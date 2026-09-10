import { collection, onSnapshot, doc, setDoc, deleteDoc } from 'firebase/firestore';
import { db } from './firebaseConfig';

export type GlobalPothole = {
  id: string;
  latitude: number;
  longitude: number;
  severity: 'low' | 'medium' | 'high';
  confidence: number;
  timestamp: number;
};

/**
 * Connects to Firebase Firestore and subscribes to a real-time stream of potholes.
 */
export const listenForGlobalPotholes = (callback: (data: GlobalPothole[]) => void) => {
  try {
    const potholesRef = collection(db, 'potholes');
    
    // onSnapshot sets up a real-time listener that fires every time the 'potholes' collection changes
    const unsubscribe = onSnapshot(potholesRef, (snapshot) => {
      const potholes: GlobalPothole[] = [];
      snapshot.forEach((doc) => {
        potholes.push(doc.data() as GlobalPothole);
      });
      // Sort by timestamp descending so newest are first
      potholes.sort((a, b) => b.timestamp - a.timestamp);
      callback(potholes);
    }, (error) => {
      console.log('[Firebase] Real-time listener error (Is your config correct?):', error);
      // Fallback: send empty array if config is invalid so app doesn't crash
      callback([]);
    });

    return unsubscribe;
  } catch (error) {
    console.log('[Firebase] Initialization error. Did you add your config to firebaseConfig.ts?');
    callback([]);
    return () => {}; // Dummy unsubscribe
  }
};

/**
 * Uploads a newly detected pothole to the global cloud database.
 */
export const syncPothole = async (pothole: GlobalPothole): Promise<void> => {
  try {
    // We use the pothole ID as the document ID to prevent duplicates
    const docRef = doc(db, 'potholes', pothole.id);
    await setDoc(docRef, pothole);
    console.log(`[Firebase] Pothole ${pothole.id} synced globally!`);
  } catch (error) {
    console.log('[Firebase] Upload failed (Waiting for real config). Error:', error);
  }
};

/**
 * The Auto-Healing Logic: Removes a pothole from the global cloud database
 * if a driver verifies that it has been fixed.
 */
export const removeFixedPothole = async (id: string): Promise<void> => {
  try {
    const docRef = doc(db, 'potholes', id);
    await deleteDoc(docRef);
    console.log(`[Firebase] Pothole ${id} verified as fixed and removed globally!`);
  } catch (error) {
    console.log('[Firebase] Delete failed. Error:', error);
  }
};
