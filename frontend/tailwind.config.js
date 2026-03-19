/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bmw: {
          cyan: '#00CCFF',
          navy: '#003366',
          red: '#FF0000',
        },
        dashboard: {
          bg: '#0F172A', // Slate 900
          card: '#1E293B', // Slate 800
          border: '#334155', // Slate 700
          text: '#F8FAFC', // Slate 50
          muted: '#94A3B8', // Slate 400
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}
