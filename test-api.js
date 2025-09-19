#!/usr/bin/env node

import http from 'http';

console.log('🧪 Testing API endpoints...\n');

// Test backend health
const testBackend = () => {
  return new Promise((resolve, reject) => {
    const req = http.get('http://localhost:8000/health', (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          const result = JSON.parse(data);
          console.log('✅ Backend API:', result.status);
          resolve(true);
        } catch (e) {
          console.log('❌ Backend API: Invalid response');
          reject(e);
        }
      });
    });

    req.on('error', (e) => {
      console.log('❌ Backend API: Connection failed');
      reject(e);
    });

    req.setTimeout(5000, () => {
      console.log('❌ Backend API: Timeout');
      reject(new Error('Timeout'));
    });
  });
};

// Test frontend
const testFrontend = () => {
  return new Promise((resolve, reject) => {
    const req = http.get('http://localhost:3000', (res) => {
      console.log('✅ Frontend:', `Status ${res.statusCode}`);
      resolve(true);
    });

    req.on('error', (e) => {
      console.log('❌ Frontend: Connection failed');
      reject(e);
    });

    req.setTimeout(5000, () => {
      console.log('❌ Frontend: Timeout');
      reject(new Error('Timeout'));
    });
  });
};

// Run tests
async function runTests() {
  try {
    await testBackend();
    await testFrontend();

    console.log('\n🎉 All tests passed!');
    console.log('\n📱 Your app is ready:');
    console.log('   Frontend: http://localhost:3000');
    console.log('   Backend API: http://localhost:8000');
    console.log('   API Docs: http://localhost:8000/docs');

  } catch (error) {
    console.log('\n❌ Some tests failed. Make sure both servers are running:');
    console.log('   Backend: cd backend && source venv/bin/activate && uvicorn main:app --reload');
    console.log('   Frontend: npm run dev');
  }
}

runTests();
