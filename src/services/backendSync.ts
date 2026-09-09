export type GlobalPothole = {
  id: string;
  latitude: number;
  longitude: number;
  severity: 'low' | 'medium' | 'high';
  confidence: number;
  timestamp: number;
};

// Simulated Cloud Database (Global State)
// We populate it with a few "crowdsourced" potholes on the route so the user can test the auto-heal feature.
let cloudDatabase: GlobalPothole[] = [
  { id: 'global_mock_1', latitude: 12.9150, longitude: 74.8550, severity: 'high', confidence: 0.95, timestamp: Date.now() - 100000 },
  { id: 'global_mock_2', latitude: 12.9180, longitude: 74.8520, severity: 'medium', confidence: 0.85, timestamp: Date.now() - 200000 },
];

let listeners: ((data: GlobalPothole[]) => void)[] = [];

const notifyListeners = () => {
  listeners.forEach(listener => listener([...cloudDatabase]));
};

/**
 * Simulates connecting to Firebase/AWS and subscribing to a real-time stream of nearby potholes.
 */
export const listenForGlobalPotholes = (callback: (data: GlobalPothole[]) => void) => {
  listeners.push(callback);
  
  // Simulate network delay for initial fetch
  setTimeout(() => {
    callback([...cloudDatabase]);
  }, 800);

  // Return unsubscribe function
  return () => {
    listeners = listeners.filter(l => l !== callback);
  };
};

/**
 * Uploads a newly detected pothole to the global cloud database.
 */
export const syncPothole = async (pothole: GlobalPothole): Promise<void> => {
  return new Promise(resolve => {
    setTimeout(() => {
      cloudDatabase.push(pothole);
      notifyListeners();
      resolve();
    }, 400); // Simulate 400ms network latency
  });
};

/**
 * The Auto-Healing Logic: Removes a pothole from the global cloud database
 * if a driver verifies that it has been fixed.
 */
export const removeFixedPothole = async (id: string): Promise<void> => {
  return new Promise(resolve => {
    setTimeout(() => {
      cloudDatabase = cloudDatabase.filter(p => p.id !== id);
      notifyListeners();
      console.log(`[CloudSync] Pothole ${id} has been verified as fixed and removed globally.`);
      resolve();
    }, 300); // Simulate 300ms network latency
  });
};
