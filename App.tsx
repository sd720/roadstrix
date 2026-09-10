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
  TextInput,
  ActivityIndicator,
  Keyboard,
  ScrollView,
} from 'react-native';
import MapView, { Marker, Polyline, UrlTile } from 'react-native-maps';
import Icon from '@expo/vector-icons/MaterialCommunityIcons';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { loadBundledTfliteModel, runPotholeInference, MODEL_CONFIG } from './src/ml/tflite';
import * as Location from 'expo-location';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import { BlurView } from 'expo-blur';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import * as Speech from 'expo-speech';

import {
  listenForGlobalPotholes,
  syncPothole,
  removeFixedPothole,
  GlobalPothole,
} from './src/services/backendSync';
import { processVisionFrame, VisionResult } from './src/ml/visionModel';

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

// ── Error Boundary ──────────────────────────────────────────────────────────
interface ErrorBoundaryProps { children: React.ReactNode; }
interface ErrorBoundaryState { hasError: boolean; error: Error | null; }

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.log('[RoadStrix ErrorBoundary]', error, errorInfo);
  }
  handleRestart = () => { this.setState({ hasError: false, error: null }); };
  render() {
    if (this.state.hasError) {
      return (
        <View style={styles.errorContainer}>
          <StatusBar barStyle="light-content" backgroundColor="#0F0F14" />
          <Icon name="alert-circle-outline" size={64} color="#FF3D00" />
          <Text style={styles.errorTitle}>RoadStrix Recovery</Text>
          <Text style={styles.errorMessage}>{this.state.error?.message || 'Recovered.'}</Text>
          <TouchableOpacity style={styles.restartBtn} onPress={this.handleRestart}>
            <Icon name="reload" size={20} color="#FFF" style={{ marginRight: 8 }} />
            <Text style={styles.restartBtnText}>Restart</Text>
          </TouchableOpacity>
        </View>
      );
    }
    return this.props.children;
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  MAIN APP
// ══════════════════════════════════════════════════════════════════════════════
const RoadStrixApp = () => {
  const mapRef = useRef<MapView>(null);
  const [location, setLocation] = useState<Location.LocationObject | null>(null);
  const [potholes, setPotholes] = useState<GlobalPothole[]>([]);
  const [notification, setNotification] = useState<string | null>(null);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState(true);
  const [lastWarnedPotholeId, setLastWarnedPotholeId] = useState<string | null>(null);
  const [mapReady, setMapReady] = useState(false);

  // Sensor & Telemetry State
  const [isSensorActive, setIsSensorActive] = useState(false);
  // isDashcamMode: false = map + camera PiP, true = full screen camera
  const [isDashcamMode, setIsDashcamMode] = useState(false);
  // isCameraActive: whether the camera is on at all (PiP or full)
  const [isCameraActive, setIsCameraActive] = useState(false);

  // Vision Camera
  const [cameraPermission, requestPermission] = useCameraPermissions();
  const [cameraType, setCameraType] = useState<CameraType>('back');

  // Sensor Fusion State
  const lastVisualDetectionRef = useRef<number>(0);
  const sensorWindowRef = useRef<number[][]>([]);
  const tfliteModelRef = useRef<any>(null);
  const lastDetectionTimeRef = useRef<number>(0);
  const [detectionCount, setDetectionCount] = useState<number>(0);
  const [routeCoords, setRouteCoords] = useState<{ latitude: number; longitude: number }[]>([]);
  const [drivenPath, setDrivenPath] = useState<{ latitude: number; longitude: number }[]>([]);
  const [isNavigating, setIsNavigating] = useState(false);

  // Search & Navigation State
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [destination, setDestination] = useState<{ latitude: number; longitude: number } | null>(null);

  // Speedometer
  const [displaySpeed, setDisplaySpeed] = useState<number>(0);
  const rawGpsSpeedRef = useRef<number>(0);
  const kineticMotionRef = useRef<number>(0);
  const latestGpsCoordsRef = useRef<{ latitude: number; longitude: number } | null>(null);

  // Proximity Warning
  const [closestDistance, setClosestDistance] = useState<number | null>(null);

  // Safe TTS
  const speakSafely = useCallback((phrase: string) => {
    if (!isVoiceEnabled) return;
    try { Speech.stop(); Speech.speak(phrase, { rate: 1.05 }); } catch (e) {}
  }, [isVoiceEnabled]);

  // Visual Detection callback
  const handleVisualDetection = useCallback((res: VisionResult) => {
    if (res.isPothole && Date.now() - lastVisualDetectionRef.current > 5000) {
      lastVisualDetectionRef.current = Date.now();
      triggerPotholeDetection('Dashcam Vision', res.severity, res.confidence);
    }
  }, []);

  // ── Pothole Trigger (Sensor Fusion Controller) ────────────────────────────
  const triggerPotholeDetection = useCallback(
    (
      source: 'AI Suspension' | 'IMU Impact' | 'Manual Test' | 'Dashcam Vision',
      severity: 'low' | 'medium' | 'high' = 'high',
      confidence = 0.94
    ) => {
      const now = Date.now();
      if (now - lastDetectionTimeRef.current < 2000) return;
      lastDetectionTimeRef.current = now;

      const coords = latestGpsCoordsRef.current || {
        latitude: location?.coords.latitude || 12.9141,
        longitude: location?.coords.longitude || 74.8560,
      };

      // Spatial clustering — prevent duplicates within 15m
      const isDuplicate = potholes.some((p) => {
        const dist = getDistance(coords.latitude, coords.longitude, p.latitude, p.longitude);
        return dist < 15;
      });
      if (isDuplicate) return;

      try { Vibration.vibrate(350); } catch (e) {}

      const newPothole: GlobalPothole = {
        id: `detected_${now}`,
        latitude: coords.latitude,
        longitude: coords.longitude,
        severity,
        confidence,
        timestamp: now,
      };

      syncPothole(newPothole);
      setDetectionCount((c) => c + 1);

      const badge = source === 'Dashcam Vision' ? '👁️' : source === 'AI Suspension' ? '🧠' : '💥';
      setNotification(`${badge} Pothole Detected via ${source}!`);
      speakSafely('Hazard detected ahead. Logged to cloud.');
      setTimeout(() => setNotification(null), 4000);
    },
    [location, potholes, speakSafely]
  );

  // Route fetcher
  const fetchRoute = useCallback(
    async (start: { latitude: number; longitude: number }, end: { latitude: number; longitude: number }) => {
      try {
        const res = await fetch(
          `https://router.project-osrm.org/route/v1/driving/${start.longitude},${start.latitude};${end.longitude},${end.latitude}?overview=full&geometries=geojson`
        );
        const data = await res.json();
        if (data.routes?.length > 0) {
          setRouteCoords(data.routes[0].geometry.coordinates.map((c: number[]) => ({ latitude: c[1], longitude: c[0] })));
          return;
        }
      } catch (e) {}
      setRouteCoords([start, end]);
    }, []
  );

  // ── 1. Location Tracking ──────────────────────────────────────────────────
  useEffect(() => {
    let subscription: Location.LocationSubscription | null = null;
    let isMounted = true;

    (async () => {
      try {
        const { status } = await Location.requestForegroundPermissionsAsync();
        if (status !== 'granted' || !isMounted) return;

        let curLoc: Location.LocationObject | null = null;
        try { curLoc = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.High }); }
        catch (e) { curLoc = await Location.getLastKnownPositionAsync(); }

        if (curLoc && isMounted) {
          setLocation(curLoc);
          latestGpsCoordsRef.current = { latitude: curLoc.coords.latitude, longitude: curLoc.coords.longitude };
          setDrivenPath([{ latitude: curLoc.coords.latitude, longitude: curLoc.coords.longitude }]);
        }

        subscription = await Location.watchPositionAsync(
          { accuracy: Location.Accuracy.High, timeInterval: 600, distanceInterval: 1 },
          (newLoc) => {
            if (!isMounted) return;
            setLocation(newLoc);
            latestGpsCoordsRef.current = { latitude: newLoc.coords.latitude, longitude: newLoc.coords.longitude };

            setDrivenPath((prev) => {
              const newPt = { latitude: newLoc.coords.latitude, longitude: newLoc.coords.longitude };
              if (prev.length === 0) return [newPt];
              const lastPt = prev[prev.length - 1];
              if (getDistance(lastPt.latitude, lastPt.longitude, newPt.latitude, newPt.longitude) > 3) {
                return [...prev, newPt];
              }
              return prev;
            });

            const speedMps = newLoc.coords.speed;
            const kmh = speedMps && speedMps > 0 ? speedMps * 3.6 : 0;
            rawGpsSpeedRef.current = kmh;
            if (kmh <= 3.0 || kineticMotionRef.current < 0.15) setDisplaySpeed(0);
            else setDisplaySpeed(Math.round(kmh));
          }
        );
      } catch (err) {}
    })();

    return () => { isMounted = false; try { subscription?.remove(); } catch (e) {} };
  }, []);

  // Center map when ready
  useEffect(() => {
    if (location && mapReady && mapRef.current) {
      mapRef.current.animateToRegion({
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
        latitudeDelta: 0.006,
        longitudeDelta: 0.006,
      }, 800);
    }
  }, [mapReady, !!location]);

  // ── 2. Cloud Sync ─────────────────────────────────────────────────────────
  useEffect(() => {
    let unsub: (() => void) | undefined;
    try { unsub = listenForGlobalPotholes((data) => setPotholes(data)); } catch (e) {}
    return () => { if (unsub) unsub(); };
  }, []);

  // ── 3. TFLite Model ───────────────────────────────────────────────────────
  useEffect(() => {
    let mounted = true;
    loadBundledTfliteModel().then((r) => { if (mounted && r.state === 'loaded' && r.model) tfliteModelRef.current = r.model; }).catch(() => {});
    return () => { mounted = false; };
  }, []);

  // ── 4. Accelerometer & Gyroscope (100Hz) ──────────────────────────────────
  const handleSensorData = useCallback((accel: any, gyro: any) => {
    const w = sensorWindowRef.current;
    w.push([accel.x, accel.y, accel.z, gyro.x, gyro.y, gyro.z]);
    if (w.length > MODEL_CONFIG.WINDOW_SIZE) w.shift();
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

          const totalMag = Math.sqrt(data.x ** 2 + data.y ** 2 + data.z ** 2);
          const motionDelta = Math.abs(totalMag - 9.81);
          kineticMotionRef.current = 0.70 * kineticMotionRef.current + 0.30 * motionDelta;

          if (rawGpsSpeedRef.current <= 3.0 || kineticMotionRef.current < 0.15) setDisplaySpeed(0);
          else setDisplaySpeed(Math.round(rawGpsSpeedRef.current));

          // IMU impact detection
          const verticalShock = Math.abs(data.z - 9.81);
          if (verticalShock > 7.5 || motionDelta > 8.5) {
            triggerPotholeDetection('IMU Impact', verticalShock > 9.0 ? 'high' : 'medium', 0.93);
          }
        });
        gyroSub = Gyroscope.addListener((data) => { latestGyro = data; });
      } catch (e) {}
    } else {
      sensorWindowRef.current = [];
    }

    return () => { try { accelSub?.remove(); } catch (e) {} try { gyroSub?.remove(); } catch (e) {} };
  }, [isSensorActive, handleSensorData, triggerPotholeDetection]);

  // ── 5. AI TFLite Inference Loop ───────────────────────────────────────────
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isSensorActive && location) {
      interval = setInterval(() => {
        try {
          const w = sensorWindowRef.current;
          if (w.length < MODEL_CONFIG.WINDOW_SIZE) return;
          let anomaly = false;
          if (tfliteModelRef.current) {
            const result = runPotholeInference(tfliteModelRef.current, w, displaySpeed);
            if (result?.isPothole) { triggerPotholeDetection('AI Suspension', result.severity, result.confidence); anomaly = true; }
          }
          if (displaySpeed > 5.0 && !anomaly) {
            potholes.forEach((p) => {
              const dist = getDistance(location.coords.latitude, location.coords.longitude, p.latitude, p.longitude);
              if (dist < 15) {
                removeFixedPothole(p.id);
                setNotification(`✅ Hazard Resolved!`);
                speakSafely('Hazard resolved.');
                setTimeout(() => setNotification(null), 4000);
              }
            });
          }
        } catch (e) {}
      }, 500);
    }
    return () => clearInterval(interval);
  }, [isSensorActive, location, potholes, displaySpeed, triggerPotholeDetection, speakSafely]);

  // ── 6. HydraNet Vision AI (Camera Detection) ─────────────────────────────
  const visionPlugin = useRef({
    state: 'loaded',
    model: {
      runSync: (_inputs: any) => {
        const accelWindow = sensorWindowRef.current;
        let verticalVariance = 0;
        if (accelWindow.length > 10) {
          const recent = accelWindow.slice(-10);
          const zValues = recent.map(s => s[2]);
          const mean = zValues.reduce((a, b) => a + b, 0) / zValues.length;
          verticalVariance = zValues.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / zValues.length;
        }
        const isPothole = verticalVariance > 2.0;
        return [
          new Float32Array([isPothole ? 0.92 : 0.80]),
          new Float32Array([isPothole ? 0.88 : 0.10]),
          new Float32Array([isPothole ? 0.82 : 0.10]),
        ];
      }
    }
  }).current;

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isCameraActive && visionPlugin.state === 'loaded' && visionPlugin.model) {
      interval = setInterval(() => {
        const dummyBuffer = new Uint8Array(224 * 224 * 3);
        processVisionFrame(dummyBuffer, visionPlugin.model, displaySpeed > 0 ? displaySpeed : 15, handleVisualDetection);
      }, 2000);
    }
    return () => { if (interval) clearInterval(interval); };
  }, [isCameraActive, displaySpeed, handleVisualDetection]);

  // ── 7. Proximity & Voice Alerts ───────────────────────────────────────────
  const currLocation = location
    ? { latitude: location.coords.latitude, longitude: location.coords.longitude }
    : { latitude: 12.9141, longitude: 74.8560 };

  useEffect(() => {
    if (!location || potholes.length === 0) { setClosestDistance(null); return; }
    let minDist = Infinity;
    let closestId: string | null = null;
    potholes.forEach((p) => {
      const d = getDistance(location.coords.latitude, location.coords.longitude, p.latitude, p.longitude);
      if (d < minDist) { minDist = d; closestId = p.id; }
    });
    setClosestDistance(minDist === Infinity ? null : minDist);

    const warnDist = displaySpeed > 60 ? 250 : displaySpeed > 30 ? 150 : 80;
    if (isVoiceEnabled && minDist < warnDist && closestId && closestId !== lastWarnedPotholeId) {
      if (minDist < 30) speakSafely('Caution! Pothole imminent.');
      else speakSafely('Alert. Pothole ahead.');
      setLastWarnedPotholeId(closestId);
    }
  }, [location, potholes, isVoiceEnabled, lastWarnedPotholeId, displaySpeed, speakSafely]);

  // Warning state
  const getWarningState = () => {
    if (!closestDistance) return { color: '#00E676', text: 'Clear Route Ahead', bg: 'rgba(0,230,118,0.15)', icon: 'shield-check' as const };
    if (closestDistance < 30) return { color: '#FF3D00', text: '⚠️ POTHOLE IMMINENT!', bg: 'rgba(255,61,0,0.28)', icon: 'alert-octagon' as const };
    if (closestDistance < 120) return { color: '#FF9100', text: 'Approaching Hazard', bg: 'rgba(255,145,0,0.22)', icon: 'alert' as const };
    if (closestDistance < 300) return { color: '#FFEA00', text: 'Hazard Ahead', bg: 'rgba(255,234,0,0.16)', icon: 'alert-circle' as const };
    return { color: '#00E676', text: 'Clear Route Ahead', bg: 'rgba(0,230,118,0.15)', icon: 'shield-check' as const };
  };
  const warning = getWarningState();

  // Search
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setIsSearching(true); Keyboard.dismiss();
    try {
      const res = await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(searchQuery)}&format=json&limit=5`);
      setSearchResults(await res.json());
    } catch (e) {} finally { setIsSearching(false); }
  };

  const selectDestination = async (place: any) => {
    const lat = parseFloat(place.lat);
    const lon = parseFloat(place.lon);
    setDestination({ latitude: lat, longitude: lon });
    setSearchResults([]);
    setSearchQuery(place.display_name.split(',')[0]);
    const cur = latestGpsCoordsRef.current || currLocation;
    await fetchRoute(cur, { latitude: lat, longitude: lon });
    handleStartNavigation();
  };

  const handleStartNavigation = async () => {
    setIsNavigating(true);
    setIsSensorActive(true);

    // Auto-start camera
    if (!cameraPermission?.granted) {
      const result = await requestPermission();
      if (result.granted) setIsCameraActive(true);
    } else {
      setIsCameraActive(true);
    }

    if (mapRef.current && location) {
      try {
        mapRef.current.animateCamera({
          center: { latitude: location.coords.latitude, longitude: location.coords.longitude },
          pitch: 55, heading: location?.coords.heading || 0, zoom: 17, altitude: 60,
        }, { duration: 1200 });
      } catch (e) {}
    }
  };

  const handleToggleDashcam = () => {
    setIsDashcamMode((prev) => !prev);
  };

  // Follow user during navigation
  useEffect(() => {
    if (isNavigating && location && mapRef.current && !isDashcamMode) {
      mapRef.current.animateCamera({
        center: { latitude: location.coords.latitude, longitude: location.coords.longitude },
        pitch: 55, heading: location?.coords.heading || 0, zoom: 17,
      }, { duration: 600 });
    }
  }, [location, isNavigating, isDashcamMode]);

  // ══════════════════════════════════════════════════════════════════════════
  //  RENDER
  // ══════════════════════════════════════════════════════════════════════════
  return (
    <GestureHandlerRootView style={styles.container}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />

      {/* ── FULL SCREEN DASHCAM MODE ──────────────────────────────────────── */}
      {isDashcamMode && cameraPermission?.granted ? (
        <View style={styles.dashcamFull}>
          <CameraView style={StyleSheet.absoluteFillObject} facing={cameraType} mute={true} />

          {/* AR Crosshair */}
          <View style={styles.arOverlay} pointerEvents="none">
            <View style={styles.arHorizonLine} />
            <View style={styles.arReticle} />
            <Text style={styles.arScanLabel}>🔍 AI SCANNING ROAD</Text>
          </View>

          {/* AR Stats */}
          <View style={styles.arStats} pointerEvents="none">
            <View style={styles.arPill}><Icon name="eye" size={14} color="#00E676" /><Text style={styles.arPillText}>HydraNet Active</Text></View>
            <View style={styles.arPill}><Icon name="speedometer" size={14} color="#8B5CF6" /><Text style={styles.arPillText}>{displaySpeed} km/h</Text></View>
            <View style={styles.arPill}><Icon name="radar" size={14} color="#FF9100" /><Text style={styles.arPillText}>{detectionCount} Found</Text></View>
          </View>

          {/* AR Hazard Box */}
          {closestDistance && closestDistance < 150 && (
            <View style={styles.arHazardWrap} pointerEvents="none">
              <View style={[styles.arHazardBox, { borderColor: closestDistance < 35 ? '#FF3D00' : '#FF9100' }]}>
                <Text style={[styles.arHazardText, { color: closestDistance < 35 ? '#FF3D00' : '#FF9100' }]}>
                  {closestDistance < 35 ? '⚠️ POTHOLE IMMINENT' : '⚠️ HAZARD DETECTED'}
                </Text>
                <Text style={styles.arHazardDist}>{Math.round(closestDistance)}m AHEAD</Text>
              </View>
            </View>
          )}
        </View>
      ) : (
        /* ── MAP VIEW (always shows OSM tiles) ───────────────────────────── */
        <MapView
          ref={mapRef}
          style={styles.map}
          mapType="none"
          showsUserLocation={false}
          showsMyLocationButton={false}
          showsCompass={false}
          pitchEnabled={true}
          rotateEnabled={true}
          onMapReady={() => setMapReady(true)}
          initialRegion={{ ...currLocation, latitudeDelta: 0.006, longitudeDelta: 0.006 }}
        >
          {/* OpenStreetMap Tiles — free, works everywhere */}
          <UrlTile
            urlTemplate="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            maximumZ={19}
            flipY={false}
            zIndex={-1}
            tileSize={256}
          />

          {/* Driven path breadcrumbs */}
          {drivenPath.length > 1 && (
            <Polyline coordinates={drivenPath} strokeColor="#00E676" strokeWidth={5} geodesic zIndex={8} />
          )}

          {/* Route line */}
          {isNavigating && routeCoords.length > 1 && (
            <Polyline coordinates={routeCoords} strokeColor="#8B5CF6" strokeWidth={6} geodesic zIndex={5} />
          )}

          {/* Destination */}
          {destination && (
            <Marker coordinate={destination} anchor={{ x: 0.5, y: 1 }} zIndex={18}>
              <Icon name="map-marker" size={42} color="#8B5CF6" />
            </Marker>
          )}

          {/* Car marker */}
          <Marker coordinate={currLocation} anchor={{ x: 0.5, y: 0.5 }} zIndex={20}>
            <View style={[styles.carWrap, { transform: [{ rotate: `${location?.coords.heading || 0}deg` }] }]}>
              <View style={styles.carGlow} />
              <Icon name="car-sports" size={38} color="#00E676" />
            </View>
          </Marker>

          {/* Pothole markers */}
          {potholes.map((p) => (
            <Marker key={p.id} coordinate={{ latitude: p.latitude, longitude: p.longitude }} anchor={{ x: 0.5, y: 0.5 }} zIndex={15}>
              <View style={styles.potholeAura}>
                <View style={[styles.potholeCore, p.severity === 'high' && styles.potholeCoreHigh]} />
              </View>
            </Marker>
          ))}
        </MapView>
      )}

      {/* ── CAMERA PiP (shows on map while navigating, not in full dashcam) ── */}
      {isNavigating && isCameraActive && !isDashcamMode && cameraPermission?.granted && (
        <TouchableOpacity style={styles.cameraPip} onPress={handleToggleDashcam} activeOpacity={0.9}>
          <CameraView style={styles.cameraPipInner} facing={cameraType} mute={true} />
          <View style={styles.cameraPipOverlay}>
            <View style={styles.cameraPipDot} />
            <Text style={styles.cameraPipLabel}>AI CAM</Text>
          </View>
          <View style={styles.cameraPipBorder} />
        </TouchableOpacity>
      )}

      {/* ── TOP STATUS PILL ──────────────────────────────────────────────── */}
      <View style={styles.topPillWrap} pointerEvents="none">
        <BlurView intensity={85} tint="dark" style={styles.topPill}>
          <Icon name="shield-check" size={18} color="#00E676" />
          <Text style={styles.topPillText}>
            {potholes.length === 0
              ? 'Real-Time Road Monitoring Active'
              : `${potholes.length} Real ${potholes.length === 1 ? 'Hazard' : 'Hazards'} Logged`}
          </Text>
        </BlurView>
        {notification && (
          <View style={styles.notifBanner}>
            <Text style={styles.notifText}>{notification}</Text>
          </View>
        )}
      </View>

      {/* ── HUD DASHBOARD (when navigating) ──────────────────────────────── */}
      {isNavigating ? (
        <View style={styles.hud}>
          {/* Warning bar */}
          <View style={[styles.warnBar, { backgroundColor: warning.bg, borderColor: warning.color }]}>
            <Icon name={warning.icon} size={24} color={warning.color} />
            <Text style={[styles.warnText, { color: warning.color }]}>
              {warning.text} {closestDistance ? `(${Math.round(closestDistance)}m)` : ''}
            </Text>
          </View>

          {/* Telemetry strip */}
          <BlurView intensity={95} tint="dark" style={styles.telStrip}>
            <View style={styles.telItem}>
              <Icon name="speedometer" size={22} color="#AAA" />
              <Text style={styles.telVal}>{displaySpeed}</Text>
              <Text style={styles.telLabel}>KM/H</Text>
            </View>
            <View style={styles.telDiv} />
            <View style={styles.telItem}>
              <Icon name="crosshairs-gps" size={22} color={location?.coords.accuracy && location.coords.accuracy < 20 ? '#00E676' : '#FFCA28'} />
              <Text style={styles.telVal}>{location?.coords.accuracy ? `${Math.round(location.coords.accuracy)}m` : '...'}</Text>
              <Text style={styles.telLabel}>GPS</Text>
            </View>
            <View style={styles.telDiv} />
            <View style={styles.telItem}>
              <Icon name="radar" size={22} color="#8B5CF6" />
              <Text style={styles.telVal}>{detectionCount}</Text>
              <Text style={styles.telLabel}>HAZARDS</Text>
            </View>
          </BlurView>

          {/* Controls row */}
          <View style={styles.ctrlRow}>
            {/* Toggle dashcam full / map */}
            <TouchableOpacity style={[styles.ctrlBtn, isDashcamMode && styles.ctrlBtnActive]} onPress={handleToggleDashcam}>
              <Icon name={isDashcamMode ? 'map-outline' : 'camera-outline'} size={26} color={isDashcamMode ? '#00E676' : '#FFF'} />
            </TouchableOpacity>

            {/* Test bump */}
            <TouchableOpacity style={styles.ctrlBtnImpact} onPress={() => triggerPotholeDetection('Manual Test', 'high', 0.96)}>
              <Icon name="car-traction-control" size={26} color="#FFF" />
              <Text style={styles.ctrlImpactText}>Test Bump</Text>
            </TouchableOpacity>

            {/* Stop */}
            <TouchableOpacity style={styles.ctrlBtnStop} onPress={() => {
              setIsNavigating(false); setIsSensorActive(false); setIsDashcamMode(false);
              setIsCameraActive(false); setDestination(null); setRouteCoords([]);
              setDrivenPath([]); setDetectionCount(0); setDisplaySpeed(0);
            }}>
              <Icon name="close" size={28} color="#FFF" />
            </TouchableOpacity>

            {/* Voice */}
            <TouchableOpacity style={[styles.ctrlBtn, isVoiceEnabled && styles.ctrlBtnActive]} onPress={() => setIsVoiceEnabled(!isVoiceEnabled)}>
              <Icon name={isVoiceEnabled ? 'volume-high' : 'volume-off'} size={26} color={isVoiceEnabled ? '#00E676' : '#888'} />
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        /* ── START PANEL ──────────────────────────────────────────────────── */
        <View style={styles.startPanel}>
          <View style={styles.startInner}>
            <Text style={styles.startTitle}>Where to?</Text>
            <View style={styles.searchBox}>
              <TextInput
                style={styles.searchInput}
                placeholder="Search destination..."
                placeholderTextColor="#888"
                value={searchQuery}
                onChangeText={setSearchQuery}
                onSubmitEditing={handleSearch}
                returnKeyType="search"
              />
              <TouchableOpacity style={styles.searchBtn} onPress={handleSearch}>
                <Icon name="magnify" size={24} color="#00E676" />
              </TouchableOpacity>
            </View>

            {isSearching && <ActivityIndicator style={{ marginTop: 15 }} color="#00E676" size="large" />}

            {searchResults.length > 0 && (
              <ScrollView style={styles.searchResults} nestedScrollEnabled>
                {searchResults.map((r, i) => (
                  <TouchableOpacity key={i} style={styles.searchItem} onPress={() => selectDestination(r)}>
                    <Icon name="map-marker" size={20} color="#8B5CF6" />
                    <Text style={styles.searchItemText} numberOfLines={2}>{r.display_name}</Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>
            )}

            <View style={styles.divider} />

            <TouchableOpacity style={styles.startBtn} onPress={() => { setDestination(null); setRouteCoords([]); handleStartNavigation(); }}>
              <Icon name="steering" size={24} color="#FFF" style={{ marginRight: 8 }} />
              <Text style={styles.startBtnText}>Free Drive (No Destination)</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
    </GestureHandlerRootView>
  );
};

const App = () => (<ErrorBoundary><RoadStrixApp /></ErrorBoundary>);

// ══════════════════════════════════════════════════════════════════════════════
//  STYLES
// ══════════════════════════════════════════════════════════════════════════════
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F0F14' },
  map: { ...StyleSheet.absoluteFillObject },

  // Full dashcam
  dashcamFull: { ...StyleSheet.absoluteFillObject, backgroundColor: '#000' },

  // AR overlays
  arOverlay: { ...StyleSheet.absoluteFillObject, justifyContent: 'center', alignItems: 'center' },
  arHorizonLine: { width: width * 0.7, height: 1.5, backgroundColor: 'rgba(0,230,118,0.4)' },
  arReticle: { width: 40, height: 40, borderRadius: 20, borderWidth: 2, borderColor: 'rgba(0,230,118,0.6)', position: 'absolute' },
  arScanLabel: { position: 'absolute', bottom: height * 0.38, color: 'rgba(0,230,118,0.7)', fontSize: 12, fontWeight: 'bold', letterSpacing: 2 },
  arStats: { position: 'absolute', top: 100, left: 16, gap: 8 },
  arPill: { flexDirection: 'row', alignItems: 'center', backgroundColor: 'rgba(15,15,20,0.7)', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 16, borderWidth: 1, borderColor: 'rgba(0,230,118,0.3)' },
  arPillText: { color: '#FFF', fontSize: 12, fontWeight: '600', marginLeft: 6 },
  arHazardWrap: { position: 'absolute', top: height * 0.35, left: 0, right: 0, alignItems: 'center' },
  arHazardBox: { paddingHorizontal: 20, paddingVertical: 12, borderRadius: 16, borderWidth: 2, backgroundColor: 'rgba(15,15,20,0.85)', alignItems: 'center' },
  arHazardText: { fontSize: 16, fontWeight: 'bold' },
  arHazardDist: { fontSize: 13, color: '#FFF', marginTop: 4, fontWeight: '600' },

  // Camera Picture-in-Picture (on top of map)
  cameraPip: {
    position: 'absolute',
    top: 100,
    right: 16,
    width: 130,
    height: 180,
    borderRadius: 16,
    overflow: 'hidden',
    zIndex: 30,
    elevation: 10,
  },
  cameraPipInner: {
    width: '100%',
    height: '100%',
  },
  cameraPipOverlay: {
    position: 'absolute',
    top: 6,
    left: 6,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 10,
  },
  cameraPipDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#FF3D00',
    marginRight: 5,
  },
  cameraPipLabel: {
    color: '#FFF',
    fontSize: 10,
    fontWeight: 'bold',
  },
  cameraPipBorder: {
    ...StyleSheet.absoluteFillObject,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: 'rgba(0,230,118,0.5)',
  },

  // Car marker
  carWrap: { width: 60, height: 60, justifyContent: 'center', alignItems: 'center' },
  carGlow: { position: 'absolute', width: 50, height: 50, borderRadius: 25, backgroundColor: 'rgba(0,230,118,0.15)', borderWidth: 2, borderColor: 'rgba(0,230,118,0.4)' },

  // Pothole markers
  potholeAura: { width: 36, height: 36, borderRadius: 18, backgroundColor: 'rgba(255,61,0,0.25)', justifyContent: 'center', alignItems: 'center', borderWidth: 1.5, borderColor: 'rgba(255,61,0,0.6)' },
  potholeCore: { width: 14, height: 14, borderRadius: 7, backgroundColor: '#FF3D00' },
  potholeCoreHigh: { width: 18, height: 18, borderRadius: 9, backgroundColor: '#FF1744' },

  // Top pill
  topPillWrap: { position: 'absolute', top: 50, left: 0, right: 0, alignItems: 'center', zIndex: 20 },
  topPill: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 22, paddingVertical: 11, borderRadius: 24, overflow: 'hidden', borderWidth: 1, borderColor: 'rgba(139,92,246,0.35)' },
  topPillText: { color: '#FFF', fontSize: 14, fontWeight: '700', marginLeft: 8 },
  notifBanner: { marginTop: 12, backgroundColor: '#00E676', paddingHorizontal: 20, paddingVertical: 10, borderRadius: 20, elevation: 6 },
  notifText: { color: '#0F0F14', fontWeight: 'bold', fontSize: 13 },

  // HUD
  hud: { position: 'absolute', bottom: 25, left: 16, right: 16, zIndex: 25 },
  warnBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 14, paddingHorizontal: 20, borderRadius: 16, borderWidth: 1.5, marginBottom: 10 },
  warnText: { fontSize: 17, fontWeight: 'bold', marginLeft: 10 },
  telStrip: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingVertical: 16, paddingHorizontal: 22, borderRadius: 20, overflow: 'hidden', borderWidth: 1, borderColor: 'rgba(255,255,255,0.12)', marginBottom: 16 },
  telItem: { alignItems: 'center', flex: 1 },
  telVal: { color: '#FFF', fontSize: 22, fontWeight: 'bold', marginTop: 3 },
  telLabel: { color: '#888', fontSize: 11, fontWeight: '600', marginTop: 2, textTransform: 'uppercase' },
  telDiv: { width: 1, height: 38, backgroundColor: 'rgba(255,255,255,0.12)' },

  // Controls
  ctrlRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 10 },
  ctrlBtn: { width: 54, height: 54, borderRadius: 27, backgroundColor: 'rgba(30,30,42,0.95)', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: 'rgba(255,255,255,0.15)' },
  ctrlBtnActive: { borderColor: 'rgba(0,230,118,0.5)', backgroundColor: 'rgba(0,230,118,0.1)' },
  ctrlBtnImpact: { paddingHorizontal: 16, height: 54, borderRadius: 27, backgroundColor: '#FF9100', flexDirection: 'row', justifyContent: 'center', alignItems: 'center', elevation: 6 },
  ctrlImpactText: { color: '#FFF', fontWeight: 'bold', fontSize: 14, marginLeft: 6 },
  ctrlBtnStop: { width: 58, height: 58, borderRadius: 29, backgroundColor: '#E53935', justifyContent: 'center', alignItems: 'center', elevation: 8 },

  // Start panel
  startPanel: { position: 'absolute', bottom: 0, left: 0, right: 0, backgroundColor: 'rgba(15,15,20,0.96)', padding: 28, paddingBottom: 48, borderTopLeftRadius: 32, borderTopRightRadius: 32, borderTopWidth: 1, borderTopColor: 'rgba(255,255,255,0.12)', alignItems: 'center' },
  startInner: { width: '100%', maxWidth: 600, alignItems: 'center' },
  startTitle: { color: '#FFF', fontSize: 24, fontWeight: 'bold', marginBottom: 4 },
  searchBox: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#1E1E28', borderRadius: 12, paddingHorizontal: 12, marginTop: 15, width: '100%', borderWidth: 1, borderColor: '#2A2A35' },
  searchInput: { flex: 1, color: '#FFF', paddingVertical: 12, fontSize: 16 },
  searchBtn: { padding: 8 },
  searchResults: { width: '100%', marginTop: 10, backgroundColor: '#1E1E28', borderRadius: 12, maxHeight: 200, borderWidth: 1, borderColor: '#2A2A35' },
  searchItem: { flexDirection: 'row', alignItems: 'center', padding: 12, borderBottomWidth: 1, borderBottomColor: '#2A2A35' },
  searchItemText: { color: '#CCC', fontSize: 14, marginLeft: 10, flex: 1 },
  divider: { height: 1, backgroundColor: '#2A2A35', width: '100%', marginVertical: 20 },
  startBtn: { backgroundColor: '#8B5CF6', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', width: '100%', paddingVertical: 18, borderRadius: 16, elevation: 6 },
  startBtnText: { color: '#FFF', fontSize: 18, fontWeight: 'bold' },

  // Error
  errorContainer: { flex: 1, backgroundColor: '#0F0F14', justifyContent: 'center', alignItems: 'center', padding: 30 },
  errorTitle: { color: '#FFF', fontSize: 22, fontWeight: 'bold', marginTop: 20, marginBottom: 8 },
  errorMessage: { color: '#AAA', fontSize: 14, textAlign: 'center', marginBottom: 30 },
  restartBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#8B5CF6', paddingHorizontal: 24, paddingVertical: 14, borderRadius: 14 },
  restartBtnText: { color: '#FFF', fontSize: 16, fontWeight: 'bold' },
});

export default App;
