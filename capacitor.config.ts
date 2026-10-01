import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.matchday.predictions',
  appName: 'Matchday',
  webDir: 'www',
  server: {
    androidScheme: 'https'
  }
};

export default config;
