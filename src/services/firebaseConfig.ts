import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

// TODO: Replace this with your actual Firebase config from the Firebase Console
const firebaseConfig = {
  apiKey: "AIzaSyDldSs7OJTUNg17823aiLxNIzMrET_UtMc",
  authDomain: "roadstrix.firebaseapp.com",
  projectId: "roadstrix",
  storageBucket: "roadstrix.firebasestorage.app",
  messagingSenderId: "641069219386",
  appId: "1:641069219386:web:f5c1768bf5c53c2aceff03",
  measurementId: "G-6G47SD1WVN"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Cloud Firestore and get a reference to the service
export const db = getFirestore(app);
