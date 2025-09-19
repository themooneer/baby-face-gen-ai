/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        baby: {
          pink: '#F8BBD9',
          blue: '#B8E6E6',
          yellow: '#FFF2CC',
          purple: '#E6CCFF',
        }
      },
      fontFamily: {
        'cute': ['Comic Sans MS', 'cursive'],
      }
    },
  },
  plugins: [],
}
