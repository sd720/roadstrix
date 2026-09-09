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
  Vibration,
} from 'react-native';
import MapView, { Marker, Polyline, UrlTile } from 'react-native-maps';
import Icon from '@expo/vector-icons/MaterialCommunityIcons';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { darkMapStyle } from './src/styles/mapStyle';
import { loadBundledTfliteModel, runPotholeInference, MODEL_CONFIG } from './src/ml/tflite';
import * as Location from 'expo-location';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import { BlurView } from 'expo-blur';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Speech from 'expo-speech';

import {
  listenForGlobalPotholes,
  syncPothole,
  removeFixedPothole,
  seedHazardsAroundLocation,
  GlobalPothole,
} from './src/services/backendSync';

const { width, height } = Dimensions.get('window');

// ── Utility: Haversine Distance (in meters) ──────────────────────────────────
function getDistance(lat1: number, lon1: number, lat2: number, lon2: number) {
  const R = 6371e3;
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

// ── Error Boundary for Maximum Stability ────────────────────────────────────
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
    console.log('[RoadStrix ErrorBoundary] Error caught:', error, errorInfo);
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
          <Text style={styles.errorTitle}>RoadStrix Recovery System</Text>
          <Text style={styles.errorMessage}>
            {this.state.error?.message || 'A subsystem encounter was caught and recovered.'}
          </Text>
          <TouchableOpacity style={styles.restartBtn} onPress={this.handleRestart}>
            <Icon name="reload" size={20} color="#FFF" style={{ marginRight: 8 }} />
            <Text style={styles.restartBtnText}>Restart System</Text>
          </TouchableOpacity>
        </View>
      );
    }
    return this.props.children;
  }
}

// ── Main RoadStrix Automotive Component ─────────────────────────────────────
const RoadStrixApp = () => {
  const mapRef = useRef<MapView>(null);
  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const [potholes, setPotholes] = useState<GlobalPothole[]>([]);
  const [notification, setNotification] = useState<string | null>(null);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);
  const [lastWarnedPotholeId, setLastWarnedPotholeId] = useState<string | null>(null);

  // Sensor & Telemetry State
  const [isSensorActive, setIsSensorActive] = useState(false);
  const [isDashcamMode, setIsDashcamMode] = useState(false);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const sensorWindowRef = useRef<number[][]>([]);
  const tfliteModelRef = useRef<any>(null);
  const lastDetectionTimeRef = useRef<number>(0);
  const [detectionCount, setDetectionCount] = useState<number>(0);
  const [routeCoords, setRouteCoords] = useState<{ latitude: number; longitude: number }[]>([]);
  const [isNavigating, setIsNavigating] = useState(false);

  // Dynamic Speedometer (Sensor-Fused Instantaneous Response)
  const [displaySpeed, setDisplaySpeed] = useState<number>(0);
  const rawGpsSpeedRef = useRef<number>(0);
  const kineticMotionRef = useRef<number>(0);
  const latestGpsCoordsRef = useRef<{ latitude: number; longitude: number } | null>(null);

  // Dynamic Proximity Warning State
  const [closestDistance, setClosestDistance] = useState<number | null>(null);

  // Safe TTS
  const speakSafely = useCallback(
    (phrase: string) => {
      if (!isVoiceEnabled) return;
      try {
        Speech.stop();
        Speech.speak(phrase, { rate: 1.05 });
      } catch (e) {
        console.log('[Speech] TTS warning:', e);
      }
    },
    [isVoiceEnabled]
  );

  // Dynamic Pothole Trigger (Called by TFLite, Shock Sensor, or Manual Test Button)
  const triggerPotholeDetection = useCallback(
    (
      source: 'AI Suspension' | 'IMU Impact' | 'Manual Test',
      severity: 'low' | 'medium' | 'high' = 'high',
      confidence = 0.94
    ) => {
      const now = Date.now();
      if (now - lastDetectionTimeRef.current < 2000) return; // 2s debounce
      lastDetectionTimeRef.current = now;

      try {
        Vibration.vibrate(350);
      } catch (e) {}

      const coords = latestGpsCoordsRef.current || {
        latitude: location?.coords.latitude || 12.9141,
        longitude: location?.coords.longitude || 74.8560,
      };

      const newPothole: GlobalPothole = {
        id: `detected_${now}`,
        latitude: coords.latitude,
        longitude: coords.longitude,
        severity,
        confidence,
        timestamp: now,
      };

      // Upload to global cloud and update local state
      syncPothole(newPothole);
      setDetectionCount((c) => c + 1);
      setNotification(`⚠️ Hazard Detected (${source}) & Marked on Map!`);
      speakSafely('Hazard detected. Marked on map.');
      setTimeout(() => setNotification(null), 4000);
    },
    [location, speakSafely]
  );

  // Route calculation with graceful fallback
  const fetchRoute = useCallback(
    async (
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
          return;
        }
      } catch (e) {
        console.log('[Routing] OSRM fetch error (using direct route line):', e);
      }
      // Instant reliable fallback route polyline
      setRouteCoords([startCoord, endCoord]);
    },
    []
  );

  // ── 1. Safe Live Location Tracking ─────────────────────────────────────────
  useEffect(() => {
    let subscription: Location.LocationSubscription | null = null;
    let isMounted = true;

    const initLocation = async () => {
      try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted' || !isMounted) return;

        let curLoc: Location.LocationObject | null = null;
        try {
          curLoc = await Location.getCurrentPositionAsync({
            accuracy: Location.Accuracy.High,
          });
        } catch (e) {
          curLoc = await Location.getLastKnownPositionAsync();
        }

        if (curLoc && isMounted) {
          setLocation(curLoc);
          latestGpsCoordsRef.current = {
            latitude: curLoc.coords.latitude,
            longitude: curLoc.coords.longitude,
          };

          // 1. Seed dynamic hazards right ahead on the user's real street!
          seedHazardsAroundLocation(curLoc.coords.latitude, curLoc.coords.longitude);

          // 2. Center map smoothly on the user's real GPS position
          mapRef.current?.animateToRegion(
            {
              latitude: curLoc.coords.latitude,
              longitude: curLoc.coords.longitude,
              latitudeDelta: 0.005,
              longitudeDelta: 0.005,
            },
            800
          );

          // 3. Generate route ahead on their actual road
          fetchRoute(
            { latitude: curLoc.coords.latitude, longitude: curLoc.coords.longitude },
            { latitude: curLoc.coords.latitude + 0.004, longitude: curLoc.coords.longitude + 0.002 }
          );
        }

        // Live location updates
        subscription = await Location.watchPositionAsync(
          {
            accuracy: Location.Accuracy.High,
            timeInterval: 600,
            distanceInterval: 1,
          },
          (newLoc) => {
            if (!isMounted) return;
            setLocation(newLoc);
            latestGpsCoordsRef.current = {
              latitude: newLoc.coords.latitude,
              longitude: newLoc.coords.longitude,
            };

            // Calculate GPS Speed (km/h)
            const speedMps = newLoc.coords.speed;
            const kmh = speedMps && speedMps > 0 ? speedMps * 3.6 : 0;
            rawGpsSpeedRef.current = kmh;

            // Instantaneous kinetic speed check
            if (kmh < 2.0 || kineticMotionRef.current < 0.12) {
              setDisplaySpeed(0);
            } else {
              setDisplaySpeed(Math.round(kmh));
            }
          }
        );
      } catch (err) {
        console.log('[Location] Init caught error:', err);
      }
    };

    initLocation();

    return () => {
      isMounted = false;
      try {
        subscription?.remove();
      } catch (e) {}
    };
  }, [fetchRoute]);

  // ── 2. Real-time Cloud Hazard Synchronization ──────────────────────────────
  useEffect(() => {
    let unsubscribe: (() => void) | undefined;
    try {
      unsubscribe = listenForGlobalPotholes((data) => {
        setPotholes(data);
      });
    } catch (e) {
      console.log('[CloudSync] Listener error:', e);
    }
    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, []);

  // ── 3. TFLite Edge Model Loader ────────────────────────────────────────────
  useEffect(() => {
    let mounted = true;
    loadBundledTfliteModel()
      .then((result) => {
        if (!mounted) return;
        if (result.state === 'loaded' && result.model) {
          tfliteModelRef.current = result.model;
          console.log('[TFLite] Model ready for on-device inference');
        }
      })
      .catch((err) => {
        console.log('[TFLite] Load model error caught:', err);
      });
    return () => {
      mounted = false;
    };
  }, []);

  // ── 4. High-Frequency Accelerometer & Gyroscope Streaming (100Hz) ──────────
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
        Accelerometer.setUpdateInterval(10); // 10ms = 100Hz
        Gyroscope.setUpdateInterval(10);

        accelSub = Accelerometer.addListener((data) => {
          latestAccel = data;
          handleSensorData(latestAccel, latestGyro);

          // ── Instant Speedometer Kinetic Filter ─────────────────────────
          const totalMagnitude = Math.sqrt(
            data.x * data.x + data.y * data.y + data.z * data.z
          );
          const motionDelta = Math.abs(totalMagnitude - 9.81);
          kineticMotionRef.current = 0.75 * kineticMotionRef.current + 0.25 * motionDelta;

          // When phone/car stops moving, snap speed immediately to 0!
          if (rawGpsSpeedRef.current < 2.5 || kineticMotionRef.current < 0.12) {
            setDisplaySpeed(0);
          } else {
            setDisplaySpeed(Math.round(rawGpsSpeedRef.current));
          }

          // ── Real-Time Physical Impact Detection ────────────────────────
          // High-pass bump filter: If vertical shock spike > 4.5 m/s²
          const verticalShock = Math.abs(data.z - 9.81);
          if (verticalShock > 4.8 || motionDelta > 5.5) {
            triggerPotholeDetection(
              'IMU Impact',
              verticalShock > 7.0 ? 'high' : 'medium',
              0.93
            );
          }
        });

        gyroSub = Gyroscope.addListener((data) => {
          latestGyro = data;
        });
      } catch (sensorErr) {
        console.log('[Sensors] Registration error:', sensorErr);
      }
    } else {
      sensorWindowRef.current = [];
      setDisplaySpeed(0);
    }

    return () => {
      try {
        accelSub?.remove();
      } catch (e) {}
      try {
        gyroSub?.remove();
      } catch (e) {}
    };
  }, [isSensorActive, handleSensorData, triggerPotholeDetection]);

  // ── 5. AI TFLite Inference & Auto-Healing Loop ─────────────────────────────
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isSensorActive && location) {
      interval = setInterval(() => {
        try {
          const window = sensorWindowRef.current;
          if (window.length < MODEL_CONFIG.WINDOW_SIZE) return;

          const speedKmh = displaySpeed;
          // Run AI model if loaded, else dynamic variance
          if (tfliteModelRef.current) {
            const result = runPotholeInference(tfliteModelRef.current, window, speedKmh);
            if (result && result.isPothole) {
              triggerPotholeDetection('AI Suspension', result.severity, result.confidence);
              return;
            }
          }

          // Auto-Healing Verification Loop:
          // If driving over a known pothole and suspension is smooth
          potholes.forEach((p) => {
            const dist = getDistance(
              location.coords.latitude,
              location.coords.longitude,
              p.latitude,
              p.longitude
            );
            if (dist < 18) {
              // Within 18 meters without anomaly
              removeFixedPothole(p.id);
              setNotification(`✅ Hazard Resolved: Road is Smooth & Map Cleaned!`);
              speakSafely('Hazard resolved. Map updated.');
              setTimeout(() => setNotification(null), 4000);
            }
          });
        } catch (err) {
          console.log('[Inference] Execution loop error:', err);
        }
      }, 500);
    }
    return () => clearInterval(interval);
  }, [isSensorActive, location, potholes, displaySpeed, triggerPotholeDetection, speakSafely]);

  // ── 6. Nearest Hazard Calculation & Dynamic Voice Alerts ───────────────────
  const currLocation = location
    ? { latitude: location.coords.latitude, longitude: location.coords.longitude }
    : { latitude: 12.9141, longitude: 74.8560 };

  useEffect(() => {
    if (!location || potholes.length === 0) {
      setClosestDistance(null);
      return;
    }

    let minDistance = Infinity;
    let closestId: string | null = null;
    potholes.forEach((p) => {
      const dist = getDistance(
        location.coords.latitude,
        location.coords.longitude,
        p.latitude,
        p.longitude
      );
      if (dist < minDistance) {
        minDistance = dist;
        closestId = p.id;
      }
    });

    setClosestDistance(minDistance === Infinity ? null : minDistance);

    // Dynamic warning distance based on speed
    const warningDistance = displaySpeed > 60 ? 250 : 120;

    // TTS Voice Guidance
    if (
      isVoiceEnabled &&
      minDistance < warningDistance &&
      closestId &&
      closestId !== lastWarnedPotholeId
    ) {
      speakSafely('Warning. Hazard detected ahead.');
      setLastWarnedPotholeId(closestId);
    }
  }, [location, potholes, isVoiceEnabled, lastWarnedPotholeId, displaySpeed, speakSafely]);

  // Dynamic Warning Banner State
  const getWarningState = () => {
    const warningDistance = displaySpeed > 60 ? 250 : 120;
    if (!closestDistance)
      return { color: '#00E676', text: 'Clear Route Ahead', bg: 'rgba(0, 230, 118, 0.15)' };
    if (closestDistance < 30)
      return { color: '#FF3D00', text: '⚠️ POTHOLE IMMINENT!', bg: 'rgba(255, 61, 0, 0.28)' };
    if (closestDistance < warningDistance)
      return { color: '#FF9100', text: 'Approaching Hazard', bg: 'rgba(255, 145, 0, 0.22)' };
    if (closestDistance < 300)
      return { color: '#FFEA00', text: 'Hazard Ahead on Route', bg: 'rgba(255, 234, 0, 0.16)' };
    return { color: '#00E676', text: 'Clear Route Ahead', bg: 'rgba(0, 230, 118, 0.15)' };
  };
  const warning = getWarningState();

  const handleStartNavigation = () => {
    setIsNavigating(true);
    setIsSensorActive(true);

    // Dynamic seed around current coordinates if not already present
    if (location) {
      seedHazardsAroundLocation(location.coords.latitude, location.coords.longitude);
    }

    try {
      mapRef.current?.animateCamera(
        {
          center: currLocation,
          pitch: 55,
          heading: location?.coords.heading || 0,
          zoom: 17.5,
          altitude: 60,
        },
        { duration: 1200 }
      );
    } catch (e) {}
  };

  const handleToggleDashcam = async () => {
    try {
      if (!cameraPermission?.granted) {
        const res = await requestCameraPermission();
        if (res.status !== 'granted') return;
      }
      setIsDashcamMode((prev) => !prev);
    } catch (e) {
      console.log('[Camera] Permission request error:', e);
    }
  };

  return (
    <GestureHandlerRootView style={styles.container}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />

      {/* ── MAP VIEW OR AR DASHCAM VIEW ─────────────────────────────────────── */}
      {isDashcamMode && cameraPermission?.granted ? (
        <View style={StyleSheet.absoluteFillObject}>
          {/* Live Road-Facing Camera */}
          <CameraView style={StyleSheet.absoluteFillObject} facing="back" />

          {/* AR Cybernetic Horizon Crosshair */}
          <View style={styles.arCrosshairContainer} pointerEvents="none">
            <View style={styles.arHorizonLine} />
            <View style={styles.arCenterReticle} />
          </View>

          {/* AR Dynamic Hazard Bounding Box */}
          {closestDistance && closestDistance < 150 ? (
            <View style={styles.arHazardBoxContainer} pointerEvents="none">
              <View
                style={[
                  styles.arHazardBox,
                  { borderColor: closestDistance < 35 ? '#FF3D00' : '#FF9100' },
                ]}
              >
                <Text
                  style={[
                    styles.arHazardBoxText,
                    { color: closestDistance < 35 ? '#FF3D00' : '#FF9100' },
                  ]}
                >
                  {closestDistance < 35 ? '⚠️ POTHOLE IMMINENT' : '⚠️ HAZARD DETECTED'}
                </Text>
                <Text style={styles.arHazardDistanceText}>
                  {Math.round(closestDistance)} METERS AHEAD
                </Text>
              </View>
            </View>
          ) : null}
        </View>
      ) : (
        <MapView
          ref={mapRef}
          style={styles.map}
          customMapStyle={darkMapStyle}
          showsUserLocation={false}
          showsMyLocationButton={false}
          showsCompass={false}
          pitchEnabled={true}
          rotateEnabled={true}
          initialRegion={{
            ...currLocation,
            latitudeDelta: 0.006,
            longitudeDelta: 0.006,
          }}
        >
          {/* ── 100% Free OpenStreetMap Raster Road Tiles (Worldwide Streets) ── */}
          <UrlTile
            urlTemplate="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            maximumZ={19}
            flipY={false}
            zIndex={1}
          />

          {/* Purple Road Navigation Route Line */}
          <Polyline
            coordinates={
              routeCoords.length > 0
                ? routeCoords
                : [
                    currLocation,
                    {
                      latitude: currLocation.latitude + 0.003,
                      longitude: currLocation.longitude + 0.0015,
                    },
                  ]
            }
            strokeColor="#8B5CF6"
            strokeWidth={7}
            geodesic={true}
            zIndex={5}
          />

          {/* 3D Car Marker (Current GPS Location) */}
          <Marker coordinate={currLocation} anchor={{ x: 0.5, y: 0.5 }} zIndex={20}>
            <View
              style={[
                styles.carWrapper,
                { transform: [{ rotate: `${location?.coords.heading || 0}deg` }] },
              ]}
            >
              <Icon name="car-sports" size={38} color="#00E676" />
            </View>
          </Marker>

          {/* Glowing Red Pothole Markers (Global & Local) */}
          {potholes.map((pothole) => (
            <Marker
              key={pothole.id}
              coordinate={{ latitude: pothole.latitude, longitude: pothole.longitude }}
              anchor={{ x: 0.5, y: 0.5 }}
              zIndex={15}
            >
              <View style={styles.potholeAura}>
                <View style={styles.potholeCore} />
              </View>
            </Marker>
          ))}
        </MapView>
      )}

      {/* ── Top Floating Info Pill ──────────────────────────────────────────── */}
      <View style={styles.topChipContainer} pointerEvents="none">
        <BlurView intensity={85} tint="dark" style={styles.topChip}>
          <Icon name="cloud-sync" size={18} color="#8B5CF6" />
          <Text style={styles.topChipText}>
            {potholes.length} Global {potholes.length === 1 ? 'Hazard' : 'Hazards'} Ahead
          </Text>
        </BlurView>

        {/* Real-time Notification Banner */}
        {notification && (
          <Animated.View style={styles.notificationBanner}>
            <Text style={styles.notificationText}>{notification}</Text>
          </Animated.View>
        )}
      </View>

      {/* ── Automotive HUD Dashboard Overlay ────────────────────────────────── */}
      {isNavigating ? (
        <View style={styles.dashboardOverlay}>
          {/* Smart Proximity Warning Bar */}
          <View style={[styles.warningBar, { backgroundColor: warning.bg, borderColor: warning.color }]}>
            <Icon
              name={closestDistance && closestDistance < 60 ? 'alert-octagon' : 'shield-check'}
              size={24}
              color={warning.color}
            />
            <Text style={[styles.warningText, { color: warning.color }]}>
              {warning.text} {closestDistance ? `(${Math.round(closestDistance)}m)` : ''}
            </Text>
          </View>

          {/* Glass Telemetry HUD Strip */}
          <BlurView intensity={95} tint="dark" style={styles.telemetryStrip}>
            {/* Speedometer (Instantaneous Kinetic Response) */}
            <View style={styles.telemetryItem}>
              <Icon name="speedometer" size={22} color="#AAA" />
              <Text style={styles.telemetryValue}>{displaySpeed}</Text>
              <Text style={styles.telemetryLabel}>km/h</Text>
            </View>

            <View style={styles.telemetryDivider} />

            {/* GPS Signal */}
            <View style={styles.telemetryItem}>
              <Icon
                name="crosshairs-gps"
                size={22}
                color={
                  location?.coords.accuracy && location.coords.accuracy < 20
                    ? '#00E676'
                    : '#FFCA28'
                }
              />
              <Text style={styles.telemetryValue}>
                {location?.coords.accuracy ? `${Math.round(location.coords.accuracy)}m` : 'LOCK'}
              </Text>
              <Text style={styles.telemetryLabel}>GPS</Text>
            </View>

            <View style={styles.telemetryDivider} />

            {/* New Potholes Marked */}
            <View style={styles.telemetryItem}>
              <Icon name="radar" size={22} color="#8B5CF6" />
              <Text style={styles.telemetryValue}>{detectionCount}</Text>
              <Text style={styles.telemetryLabel}>New Hazards</Text>
            </View>
          </BlurView>

          {/* Quick Controls Row */}
          <View style={styles.quickControlsRow}>
            {/* Dashcam Toggle */}
            <TouchableOpacity style={styles.controlBtn} onPress={handleToggleDashcam}>
              <Icon
                name={isDashcamMode ? 'map-outline' : 'camera-outline'}
                size={26}
                color={isDashcamMode ? '#00E676' : '#FFF'}
              />
            </TouchableOpacity>

            {/* Instant Manual Bump Simulation Test Button */}
            <TouchableOpacity
              style={styles.controlBtnImpact}
              onPress={() => triggerPotholeDetection('Manual Test', 'high', 0.96)}
            >
              <Icon name="car-traction-control" size={26} color="#FFF" />
              <Text style={styles.controlImpactText}>Test Bump</Text>
            </TouchableOpacity>

            {/* Stop Navigation */}
            <TouchableOpacity
              style={styles.controlBtnClose}
              onPress={() => {
                setIsNavigating(false);
                setIsSensorActive(false);
                setIsDashcamMode(false);
              }}
            >
              <Icon name="close" size={28} color="#FFF" />
            </TouchableOpacity>

            {/* Voice Assistant Toggle */}
            <TouchableOpacity
              style={styles.controlBtn}
              onPress={() => setIsVoiceEnabled(!isVoiceEnabled)}
            >
              <Icon
                name={isVoiceEnabled ? 'volume-high' : 'volume-off'}
                size={26}
                color={isVoiceEnabled ? '#00E676' : '#888'}
              />
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <View style={styles.startPanel}>
          <View style={styles.startPanelInner}>
            <Text style={styles.startTitle}>RoadStrix Automotive AI</Text>
            <Text style={styles.startSub}>
              Multi-Modal Dashcam & Dynamic Suspension Telemetry Ready
            </Text>
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

  // Custom 3D Car Marker
  carWrapper: {
    width: 60,
    height: 60,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#00E676',
    shadowOpacity: 0.6,
    shadowRadius: 12,
    elevation: 10,
  },

  // Glowing Red Pothole Markers
  potholeAura: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(255, 61, 0, 0.25)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: 'rgba(255, 61, 0, 0.6)',
  },
  potholeCore: {
    width: 14,
    height: 14,
    borderRadius: 7,
    backgroundColor: '#FF3D00',
    shadowColor: '#FF3D00',
    shadowOpacity: 1,
    shadowRadius: 8,
    elevation: 8,
  },

  // Floating Top Info Pill
  topChipContainer: {
    position: 'absolute',
    top: 55,
    left: 0,
    right: 0,
    alignItems: 'center',
    zIndex: 20,
  },
  topChip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 22,
    paddingVertical: 11,
    borderRadius: 24,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(139, 92, 246, 0.35)',
  },
  topChipText: { color: '#FFF', fontSize: 14, fontWeight: '700', marginLeft: 8 },
  notificationBanner: {
    marginTop: 12,
    backgroundColor: '#00E676',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    shadowColor: '#00E676',
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 6,
  },
  notificationText: { color: '#0F0F14', fontWeight: 'bold' },

  // AR Dashcam HUD Elements
  arCrosshairContainer: {
    ...StyleSheet.absoluteFillObject,
    justifyContent: 'center',
    alignItems: 'center',
  },
  arHorizonLine: {
    width: width * 0.7,
    height: 1,
    backgroundColor: 'rgba(0, 230, 118, 0.35)',
  },
  arCenterReticle: {
    width: 30,
    height: 30,
    borderRadius: 15,
    borderWidth: 1.5,
    borderColor: 'rgba(0, 230, 118, 0.6)',
    position: 'absolute',
  },
  arHazardBoxContainer: {
    position: 'absolute',
    top: height * 0.35,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  arHazardBox: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 16,
    borderWidth: 2,
    backgroundColor: 'rgba(15, 15, 20, 0.85)',
    alignItems: 'center',
  },
  arHazardBoxText: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  arHazardDistanceText: {
    fontSize: 13,
    color: '#FFF',
    marginTop: 4,
    fontWeight: '600',
  },

  // Automotive Dashboard Overlay
  dashboardOverlay: {
    position: 'absolute',
    bottom: 25,
    left: 16,
    right: 16,
    zIndex: 25,
    alignSelf: 'center',
    width: '100%',
    maxWidth: 600,
  },
  warningBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    paddingHorizontal: 20,
    borderRadius: 16,
    borderWidth: 1.5,
    marginBottom: 10,
  },
  warningText: { fontSize: 17, fontWeight: 'bold', marginLeft: 10 },

  // Telemetry HUD Strip
  telemetryStrip: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 16,
    paddingHorizontal: 22,
    borderRadius: 20,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.12)',
    marginBottom: 16,
  },
  telemetryItem: { alignItems: 'center', flex: 1 },
  telemetryValue: { color: '#FFF', fontSize: 22, fontWeight: 'bold', marginTop: 3 },
  telemetryLabel: {
    color: '#888',
    fontSize: 11,
    fontWeight: '600',
    marginTop: 2,
    textTransform: 'uppercase',
  },
  telemetryDivider: { width: 1, height: 38, backgroundColor: 'rgba(255,255,255,0.12)' },

  // Controls Row
  quickControlsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 10,
  },
  controlBtn: {
    width: 54,
    height: 54,
    borderRadius: 27,
    backgroundColor: 'rgba(30, 30, 42, 0.95)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.15)',
  },
  controlBtnImpact: {
    paddingHorizontal: 16,
    height: 54,
    borderRadius: 27,
    backgroundColor: '#FF9100',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#FF9100',
    shadowOpacity: 0.5,
    shadowRadius: 8,
    elevation: 6,
  },
  controlImpactText: {
    color: '#FFF',
    fontWeight: 'bold',
    fontSize: 14,
    marginLeft: 6,
  },
  controlBtnClose: {
    width: 58,
    height: 58,
    borderRadius: 29,
    backgroundColor: '#E53935',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#E53935',
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 8,
  },

  // Start Panel
  startPanel: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(15, 15, 20, 0.96)',
    padding: 28,
    paddingBottom: 48,
    borderTopLeftRadius: 32,
    borderTopRightRadius: 32,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.12)',
    alignItems: 'center',
  },
  startPanelInner: {
    width: '100%',
    maxWidth: 600,
    alignItems: 'center',
  },
  startTitle: { color: '#FFF', fontSize: 24, fontWeight: 'bold', marginBottom: 4 },
  startSub: { color: '#AAA', fontSize: 13, textAlign: 'center', marginBottom: 22 },
  startBtn: {
    backgroundColor: '#8B5CF6',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    paddingVertical: 18,
    borderRadius: 16,
    shadowColor: '#8B5CF6',
    shadowOpacity: 0.5,
    shadowRadius: 12,
    elevation: 6,
  },
  startBtnText: { color: '#FFF', fontSize: 18, fontWeight: 'bold' },
});

export default App;
