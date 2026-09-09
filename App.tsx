import 'react-native-gesture-handler';
import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  Dimensions,
  Animated,
} from 'react-native';
import MapView, { Marker, Polyline } from 'react-native-maps';
import Icon from '@expo/vector-icons/MaterialCommunityIcons';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { darkMapStyle } from './src/styles/mapStyle';
import { loadBundledTfliteModel, runPotholeInference, MODEL_CONFIG } from './src/ml/tflite';
import * as Location from 'expo-location';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import { BlurView } from 'expo-blur';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Speech from 'expo-speech';

import { listenForGlobalPotholes, syncPothole, removeFixedPothole, GlobalPothole } from './src/services/backendSync';

const { width } = Dimensions.get('window');

// ── Utility: Haversine Distance ──────────────────────────────────────────────
function getDistance(lat1: number, lon1: number, lat2: number, lon2: number) {
  const R = 6371e3; // metres
  const p1 = (lat1 * Math.PI) / 180;
  const p2 = (lat2 * Math.PI) / 180;
  const dp = ((lat2 - lat1) * Math.PI) / 180;
  const dl = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dp / 2) * Math.sin(dp / 2) +
    Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) * Math.sin(dl / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

const initialDestLocation = { latitude: 12.9538, longitude: 74.8340 };

// ── Error Boundary for Crash Prevention ─────────────────────────────────────
interface ErrorBoundaryProps {
  children: React.ReactNode;
}
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.log('[RoadStrix ErrorBoundary] Caught error:', error, errorInfo);
  }

  handleRestart = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <View style={styles.errorContainer}>
          <StatusBar barStyle="light-content" backgroundColor="#0F0F14" />
          <Icon name="alert-circle-outline" size={64} color="#FF3D00" />
          <Text style={styles.errorTitle}>RoadStrix Diagnostics</Text>
          <Text style={styles.errorMessage}>
            {this.state.error?.message || 'A subsystem encounter was caught and handled safely.'}
          </Text>
          <TouchableOpacity style={styles.restartBtn} onPress={this.handleRestart}>
            <Icon name="reload" size={20} color="#FFF" style={{ marginRight: 8 }} />
            <Text style={styles.restartBtnText}>Restart Application</Text>
          </TouchableOpacity>
        </View>
      );
    }
    return this.props.children;
  }
}

// ── Main RoadStrix Component ────────────────────────────────────────────────
const RoadStrixApp = () => {
  const mapRef = useRef<MapView>(null);
  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const [potholes, setPotholes] = useState<GlobalPothole[]>([]);
  const [notification, setNotification] = useState<string | null>(null);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);
  const [lastWarnedPotholeId, setLastWarnedPotholeId] = useState<string | null>(null);

  const [isSensorActive, setIsSensorActive] = useState(false);
  const [isDashcamMode, setIsDashcamMode] = useState(false);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const sensorWindowRef = useRef<number[][]>([]);
  const tfliteModelRef = useRef<any>(null);
  const lastDetectionTimeRef = useRef<number>(0);
  const [detectionCount, setDetectionCount] = useState<number>(0);
  const [routeCoords, setRouteCoords] = useState<{ latitude: number; longitude: number }[]>([]);
  const [isNavigating, setIsNavigating] = useState(false);

  // Dynamic Warning State
  const [closestDistance, setClosestDistance] = useState<number | null>(null);

  const speakSafely = useCallback(
    (phrase: string) => {
      if (!isVoiceEnabled) return;
      try {
        Speech.speak(phrase, { rate: 1.1 });
      } catch (e) {
        console.log('[Speech] TTS speak warning:', e);
      }
    },
    [isVoiceEnabled]
  );

  const fetchRoute = async (
    startCoord: { latitude: number; longitude: number },
    endCoord: { latitude: number; longitude: number }
  ) => {
    try {
      const res = await fetch(
        `https://router.project-osrm.org/route/v1/driving/${startCoord.longitude},${startCoord.latitude};${endCoord.longitude},${endCoord.latitude}?overview=full&geometries=geojson`
      );
      const data = await res.json();
      if (data.routes && data.routes.length > 0) {
        const coords = data.routes[0].geometry.coordinates.map((c: number[]) => ({
          latitude: c[1],
          longitude: c[0],
        }));
        setRouteCoords(coords);
      }
    } catch (e) {
      console.log('Routing error:', e);
    }
  };

  // Safe Location Initialization
  useEffect(() => {
    let subscription: Location.LocationSubscription | null = null;
    let isMounted = true;

    const setupLocation = async () => {
      try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted' || !isMounted) return;

        try {
          const currentLocation = await Location.getCurrentPositionAsync({
            accuracy: Location.Accuracy.Balanced,
          });
          if (isMounted) setLocation(currentLocation);
        } catch (posErr) {
          console.log('[Location] Current pos fallback:', posErr);
          try {
            const lastKnown = await Location.getLastKnownPositionAsync();
            if (isMounted && lastKnown) setLocation(lastKnown);
          } catch (lastErr) {
            console.log('[Location] Last known pos fallback failed:', lastErr);
          }
        }

        subscription = await Location.watchPositionAsync(
          { accuracy: Location.Accuracy.High, timeInterval: 1000, distanceInterval: 1 },
          (newLoc) => {
            if (isMounted) setLocation(newLoc);
          }
        );
      } catch (err) {
        console.log('[Location] Setup caught error:', err);
      }
    };

    setupLocation();

    return () => {
      isMounted = false;
      try {
        subscription?.remove();
      } catch (e) {
        // ignore
      }
    };
  }, []);

  useEffect(() => {
    fetchRoute({ latitude: 12.9141, longitude: 74.8560 }, initialDestLocation);
  }, []);

  // ── Global Cloud Sync ──────────────────────────────────────────────────────
  useEffect(() => {
    let unsubscribe: (() => void) | undefined;
    try {
      unsubscribe = listenForGlobalPotholes((data) => {
        setPotholes(data);
      });
    } catch (syncErr) {
      console.log('[CloudSync] Setup error:', syncErr);
    }
    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, []);

  // Safe TFLite Model Loading
  useEffect(() => {
    let mounted = true;
    loadBundledTfliteModel()
      .then((result) => {
        if (!mounted) return;
        if (result.state === 'loaded' && result.model) {
          tfliteModelRef.current = result.model;
        }
      })
      .catch((err) => {
        console.log('[TFLite] loadBundledTfliteModel error caught:', err);
      });
    return () => {
      mounted = false;
    };
  }, []);

  // ── Sensor Inference Loop ──────────────────────────────────────────────────
  const handleSensorData = useCallback((accel: any, gyro: any) => {
    const window = sensorWindowRef.current;
    window.push([accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z]);
    if (window.length > MODEL_CONFIG.WINDOW_SIZE) window.shift();
  }, []);

  useEffect(() => {
    let accelSub: any = null;
    let gyroSub: any = null;
    let latestAccel = { x: 0, y: 0, z: 9.81 };
    let latestGyro = { x: 0, y: 0, z: 0 };

    if (isSensorActive) {
      try {
        Accelerometer.setUpdateInterval(10);
        Gyroscope.setUpdateInterval(10);
        accelSub = Accelerometer.addListener((data) => {
          latestAccel = data;
          handleSensorData(latestAccel, latestGyro);
        });
        gyroSub = Gyroscope.addListener((data) => {
          latestGyro = data;
        });
      } catch (sensorErr) {
        console.log('[Sensors] Listener registration error:', sensorErr);
      }
    } else {
      sensorWindowRef.current = [];
    }

    return () => {
      try {
        accelSub?.remove();
      } catch (e) {}
      try {
        gyroSub?.remove();
      } catch (e) {}
    };
  }, [isSensorActive, handleSensorData]);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isSensorActive && tfliteModelRef.current && location) {
      interval = setInterval(() => {
        try {
          const window = sensorWindowRef.current;
          if (window.length < MODEL_CONFIG.WINDOW_SIZE) return;

          const speedKmh = location.coords.speed ? Math.max(0, location.coords.speed * 3.6) : 0;
          const result = runPotholeInference(tfliteModelRef.current, window, speedKmh);

          if (result) {
            const now = Date.now();

            if (result.isPothole) {
              // Found a pothole!
              if (now - lastDetectionTimeRef.current > 2000) {
                lastDetectionTimeRef.current = now;
                const newPothole: GlobalPothole = {
                  id: now.toString(),
                  latitude: location.coords.latitude,
                  longitude: location.coords.longitude,
                  severity: result.severity,
                  confidence: result.confidence,
                  timestamp: now,
                };
                syncPothole(newPothole); // Upload to Cloud
                setDetectionCount((p) => p + 1);
              }
            } else {
              // Road is smooth. Auto-Healing Logic:
              potholes.forEach((p) => {
                const dist = getDistance(
                  location.coords.latitude,
                  location.coords.longitude,
                  p.latitude,
                  p.longitude
                );
                if (dist < 15) {
                  // Within 15 meters
                  removeFixedPothole(p.id);
                  setNotification(`✅ Hazard Resolved: Map Updated for everyone`);
                  speakSafely('Hazard resolved. Map updated.');
                  setTimeout(() => setNotification(null), 4000);
                }
              });
            }
          }
        } catch (infErr) {
          console.log('[Inference] Loop execution error:', infErr);
        }
      }, 500);
    }
    return () => clearInterval(interval);
  }, [isSensorActive, location, potholes, speakSafely]);

  // ── Nearest Pothole Calculation ─────────────────────────────────────────────
  const currLocation = location
    ? { latitude: location.coords.latitude, longitude: location.coords.longitude }
    : { latitude: 12.9141, longitude: 74.8560 };

  const speedKmh = location?.coords.speed ? Math.round(location.coords.speed * 3.6) : 0;

  useEffect(() => {
    if (!location || potholes.length === 0) return;
    let minDistance = Infinity;
    let closestId: string | null = null;
    potholes.forEach((p) => {
      const dist = getDistance(
        location.coords.latitude,
        location.coords.longitude,
        p.latitude,
        p.longitude
      );
      if (dist < minDistance && dist > 5) {
        minDistance = dist;
        closestId = p.id;
      }
    });
    setClosestDistance(minDistance === Infinity ? null : minDistance);

    // Dynamic Warning Distance based on speed
    const warningDistance = speedKmh > 80 ? 250 : 100;

    // TTS Voice Alerts
    if (isVoiceEnabled && minDistance < warningDistance && closestId && closestId !== lastWarnedPotholeId) {
      speakSafely('Warning. Hazard detected ahead.');
      setLastWarnedPotholeId(closestId);
    }
  }, [location, potholes, isVoiceEnabled, lastWarnedPotholeId, speedKmh, speakSafely]);

  // Smart Warning Logic
  const getWarningState = () => {
    const warningDistance = speedKmh > 80 ? 250 : 100;
    if (!closestDistance) return { color: '#00E676', text: 'Clear Route', bg: 'rgba(0, 230, 118, 0.15)' };
    if (closestDistance < 30) return { color: '#FF3D00', text: 'POTHOLE IMMINENT!', bg: 'rgba(255, 61, 0, 0.25)' };
    if (closestDistance < warningDistance)
      return { color: '#FF9100', text: 'Approaching Hazard', bg: 'rgba(255, 145, 0, 0.2)' };
    if (closestDistance < 300) return { color: '#FFEA00', text: 'Hazard Ahead', bg: 'rgba(255, 234, 0, 0.15)' };
    return { color: '#00E676', text: 'Clear Route', bg: 'rgba(0, 230, 118, 0.15)' };
  };
  const warning = getWarningState();

  const handleStartNavigation = () => {
    setIsNavigating(true);
    setIsSensorActive(true);
    try {
      mapRef.current?.animateCamera(
        {
          center: currLocation,
          pitch: 60,
          heading: location?.coords.heading || 0,
          zoom: 18,
          altitude: 50,
        },
        { duration: 1500 }
      );
    } catch (animErr) {
      console.log('[Map] Camera animation error:', animErr);
    }
  };

  const handleToggleDashcam = async () => {
    try {
      if (!cameraPermission?.granted) {
        const res = await requestCameraPermission();
        if (res.status !== 'granted') return;
      }
      setIsDashcamMode((prev) => !prev);
    } catch (camErr) {
      console.log('[Camera] Permission error:', camErr);
    }
  };

  return (
    <GestureHandlerRootView style={styles.container}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />

      {isDashcamMode && cameraPermission?.granted ? (
        <CameraView style={StyleSheet.absoluteFillObject} facing="back" />
      ) : (
        <MapView
          ref={mapRef}
          style={styles.map}
          customMapStyle={darkMapStyle}
          showsUserLocation={false}
          showsMyLocationButton={false}
          showsCompass={false}
          pitchEnabled={true}
          initialRegion={{ ...currLocation, latitudeDelta: 0.015, longitudeDelta: 0.012 }}
        >
          {/* Route Line */}
          <Polyline
            coordinates={routeCoords.length > 0 ? routeCoords : [currLocation, initialDestLocation]}
            strokeColor="#8B5CF6"
            strokeWidth={6}
            geodesic={true}
          />

          {/* 3D Car Icon (Current Location) */}
          <Marker coordinate={currLocation} anchor={{ x: 0.5, y: 0.5 }}>
            <View
              style={[
                styles.carWrapper,
                { transform: [{ rotate: `${location?.coords.heading || 0}deg` }] },
              ]}
            >
              <Icon name="car-sports" size={36} color="#00E676" />
            </View>
          </Marker>

          {/* Pothole Markers */}
          {potholes.map((pothole) => (
            <Marker
              key={pothole.id}
              coordinate={{ latitude: pothole.latitude, longitude: pothole.longitude }}
            >
              <View style={styles.potholeAura}>
                <View style={styles.potholeCore} />
              </View>
            </Marker>
          ))}
        </MapView>
      )}

      {/* ── Top Floating Info Chip ──────────────────────────────────────────── */}
      <View style={styles.topChipContainer} pointerEvents="none">
        <BlurView intensity={80} tint="dark" style={styles.topChip}>
          <Icon name="cloud-sync" size={18} color="#8B5CF6" />
          <Text style={styles.topChipText}>{potholes.length} Global Hazards Ahead</Text>
        </BlurView>

        {/* Auto-Healing Notification Banner */}
        {notification && (
          <Animated.View style={styles.notificationBanner}>
            <Text style={styles.notificationText}>{notification}</Text>
          </Animated.View>
        )}
      </View>

      {/* ── Mid Info Strip & Smart Warning (Automotive Dashboard Layout) ───── */}
      {isNavigating ? (
        <View style={styles.dashboardOverlay}>
          {/* Smart Warning Bar */}
          <View style={[styles.warningBar, { backgroundColor: warning.bg, borderColor: warning.color }]}>
            <Icon
              name={closestDistance && closestDistance < 100 ? 'alert-octagon' : 'shield-check'}
              size={24}
              color={warning.color}
            />
            <Text style={[styles.warningText, { color: warning.color }]}>
              {warning.text} {closestDistance ? `(${Math.round(closestDistance)}m)` : ''}
            </Text>
          </View>

          {/* Glass Telemetry Strip */}
          <BlurView intensity={90} tint="dark" style={styles.telemetryStrip}>
            <View style={styles.telemetryItem}>
              <Icon name="speedometer" size={20} color="#AAA" />
              <Text style={styles.telemetryValue}>{speedKmh}</Text>
              <Text style={styles.telemetryLabel}>km/h</Text>
            </View>
            <View style={styles.telemetryDivider} />
            <View style={styles.telemetryItem}>
              <Icon
                name="crosshairs-gps"
                size={20}
                color={
                  location?.coords.accuracy && location.coords.accuracy < 20 ? '#00E676' : '#FFCA28'
                }
              />
              <Text style={styles.telemetryValue}>GPS</Text>
              <Text style={styles.telemetryLabel}>Signal</Text>
            </View>
            <View style={styles.telemetryDivider} />
            <View style={styles.telemetryItem}>
              <Icon name="radar" size={20} color="#8B5CF6" />
              <Text style={styles.telemetryValue}>{detectionCount}</Text>
              <Text style={styles.telemetryLabel}>New</Text>
            </View>
          </BlurView>

          {/* Quick Controls */}
          <View style={styles.quickControlsRow}>
            <TouchableOpacity style={styles.controlBtn} onPress={handleToggleDashcam}>
              <Icon
                name={isDashcamMode ? 'map-outline' : 'camera-outline'}
                size={24}
                color={isDashcamMode ? '#00E676' : '#FFF'}
              />
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.controlBtnClose}
              onPress={() => {
                setIsNavigating(false);
                setIsSensorActive(false);
              }}
            >
              <Icon name="close" size={28} color="#FFF" />
            </TouchableOpacity>
            <TouchableOpacity style={styles.controlBtn} onPress={() => setIsVoiceEnabled(!isVoiceEnabled)}>
              <Icon
                name={isVoiceEnabled ? 'volume-high' : 'volume-off'}
                size={24}
                color={isVoiceEnabled ? '#00E676' : '#888'}
              />
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <View style={styles.startPanel}>
          <View style={styles.startPanelInner}>
            <Text style={styles.startTitle}>RoadStrix Navigation</Text>
            <Text style={styles.startSub}>Suspension Telemetry Ready</Text>
            <TouchableOpacity style={styles.startBtn} onPress={handleStartNavigation}>
              <Icon name="steering" size={24} color="#FFF" style={{ marginRight: 8 }} />
              <Text style={styles.startBtnText}>Start Drive</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
    </GestureHandlerRootView>
  );
};

const App = () => {
  return (
    <ErrorBoundary>
      <RoadStrixApp />
    </ErrorBoundary>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F0F14' },
  map: { ...StyleSheet.absoluteFillObject },

  // Error Boundary Fallback
  errorContainer: {
    flex: 1,
    backgroundColor: '#0F0F14',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 30,
  },
  errorTitle: {
    color: '#FFF',
    fontSize: 22,
    fontWeight: 'bold',
    marginTop: 20,
    marginBottom: 8,
  },
  errorMessage: {
    color: '#AAA',
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 30,
  },
  restartBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#8B5CF6',
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 14,
  },
  restartBtnText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },

  // Custom Map Markers
  carWrapper: {
    width: 60,
    height: 60,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#00E676',
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 8,
  },
  potholeAura: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(255, 61, 0, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 61, 0, 0.5)',
  },
  potholeCore: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#FF3D00',
    shadowColor: '#FF3D00',
    shadowOpacity: 1,
    shadowRadius: 6,
    elevation: 5,
  },

  // Floating Info Chip
  topChipContainer: {
    position: 'absolute',
    top: 55,
    left: 0,
    right: 0,
    alignItems: 'center',
    zIndex: 10,
  },
  topChip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 24,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(139, 92, 246, 0.3)',
  },
  topChipText: { color: '#FFF', fontSize: 14, fontWeight: '600', marginLeft: 8 },
  notificationBanner: {
    marginTop: 12,
    backgroundColor: '#00E676',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    shadowColor: '#00E676',
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 5,
  },
  notificationText: { color: '#0F0F14', fontWeight: 'bold' },

  // Dashboard Overlay
  dashboardOverlay: {
    position: 'absolute',
    bottom: 30,
    left: 20,
    right: 20,
    zIndex: 10,
    alignSelf: 'center',
    width: '100%',
    maxWidth: 600,
  },

  // Smart Warning Bar
  warningBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 16,
    borderWidth: 1,
    marginBottom: 12,
  },
  warningText: { fontSize: 18, fontWeight: 'bold', marginLeft: 10 },

  // Telemetry Strip
  telemetryStrip: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 16,
    paddingHorizontal: 24,
    borderRadius: 20,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
    marginBottom: 20,
  },
  telemetryItem: { alignItems: 'center', flex: 1 },
  telemetryValue: { color: '#FFF', fontSize: 20, fontWeight: 'bold', marginTop: 4 },
  telemetryLabel: {
    color: '#888',
    fontSize: 11,
    fontWeight: '600',
    marginTop: 2,
    textTransform: 'uppercase',
  },
  telemetryDivider: { width: 1, height: 40, backgroundColor: 'rgba(255,255,255,0.1)' },

  // Quick Controls
  quickControlsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
  },
  controlBtn: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: 'rgba(30, 30, 40, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  controlBtnClose: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: '#E53935',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#E53935',
    shadowOpacity: 0.4,
    shadowRadius: 10,
    elevation: 8,
  },

  // Start Panel
  startPanel: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(15, 15, 20, 0.95)',
    padding: 30,
    paddingBottom: 50,
    borderTopLeftRadius: 32,
    borderTopRightRadius: 32,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.1)',
    alignItems: 'center',
  },
  startPanelInner: {
    width: '100%',
    maxWidth: 600,
    alignItems: 'center',
  },
  startTitle: { color: '#FFF', fontSize: 24, fontWeight: 'bold', marginBottom: 4 },
  startSub: { color: '#AAA', fontSize: 14, marginBottom: 24 },
  startBtn: {
    backgroundColor: '#8B5CF6',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    paddingVertical: 18,
    borderRadius: 16,
    shadowColor: '#8B5CF6',
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 6,
  },
  startBtnText: { color: '#FFF', fontSize: 18, fontWeight: 'bold' },
});

export default App;
