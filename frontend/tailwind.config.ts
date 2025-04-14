/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'dream-blue': '#1a365d',
        'dream-gold': '#f6ad55',
      },
    },
  },
  plugins: [],
}