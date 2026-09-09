export type GlobalPothole = {
  id: string;
  latitude: number;
  longitude: number;
  severity: 'low' | 'medium' | 'high';
  confidence: number;
  timestamp: number;
};

// Simulated Cloud Database (Global State)
let cloudDatabase: GlobalPothole[] = [];
let hasSeededLocation = false;

let listeners: ((data: GlobalPothole[]) => void)[] = [];

const notifyListeners = () => {
  listeners.forEach((listener) => listener([...cloudDatabase]));
};

/**
 * Dynamically seeds crowdsourced potholes relative to the user's actual live position.
 * This guarantees hazards are immediately visible ahead on their actual road anywhere in the world!
 */
export const seedHazardsAroundLocation = (lat: number, lon: number, force = false) => {
  if (hasSeededLocation && !force) return;
  hasSeededLocation = true;

  // Generate 2 sample crowdsourced hazards ahead along the road
  cloudDatabase = [
    {
      id: `crowdsourced_${Date.now()}_1`,
      latitude: lat + 0.0009, // ~100m ahead
      longitude: lon + 0.0004,
      severity: 'high',
      confidence: 0.94,
      timestamp: Date.now() - 360000,
    },
    {
      id: `crowdsourced_${Date.now()}_2`,
      latitude: lat + 0.0022, // ~250m ahead
      longitude: lon + 0.0011,
      severity: 'medium',
      confidence: 0.88,
      timestamp: Date.now() - 720000,
    },
  ];
  notifyListeners();
  console.log(`[CloudSync] Seeded ${cloudDatabase.length} crowdsourced hazards around user's real GPS.`);
};

/**
 * Simulates connecting to Firebase/AWS and subscribing to a real-time stream of nearby potholes.
 */
export const listenForGlobalPotholes = (callback: (data: GlobalPothole[]) => void) => {
  listeners.push(callback);

  // Return current state immediately
  callback([...cloudDatabase]);

  // Return unsubscribe function
  return () => {
    listeners = listeners.filter((l) => l !== callback);
  };
};

/**
 * Uploads a newly detected pothole to the global cloud database.
 */
export const syncPothole = async (pothole: GlobalPothole): Promise<void> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Add to front so newly detected potholes take priority
      cloudDatabase.unshift(pothole);
      notifyListeners();
      resolve();
    }, 200); // 200ms latency
  });
};

/**
 * The Auto-Healing Logic: Removes a pothole from the global cloud database
 * if a driver verifies that it has been fixed.
 */
export const removeFixedPothole = async (id: string): Promise<void> => {
  return new Promise((resolve) => {
    setTimeout(() => {
      cloudDatabase = cloudDatabase.filter((p) => p.id !== id);
      notifyListeners();
      console.log(`[CloudSync] Pothole ${id} verified as fixed and removed globally.`);
      resolve();
    }, 150);
  });
};
