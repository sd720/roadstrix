export type GlobalPothole = {
  id: string;
  latitude: number;
  longitude: number;
  severity: 'low' | 'medium' | 'high';
  confidence: number;
  timestamp: number;
};

// Real-time Cloud Hazard Database (Potholes logged only via genuine detections)
let cloudDatabase: GlobalPothole[] = [];

let listeners: ((data: GlobalPothole[]) => void)[] = [];

const notifyListeners = () => {
  listeners.forEach((listener) => listener([...cloudDatabase]));
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
