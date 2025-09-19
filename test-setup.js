#!/usr/bin/env node

import fs from 'fs';
import path from 'path';

console.log('🧪 Testing Baby Face Generator Setup...\n');

// Check if required files exist
const requiredFiles = [
  'package.json',
  'vite.config.ts',
  'tsconfig.json',
  'tailwind.config.js',
  'src/App.tsx',
  'src/main.tsx',
  'src/index.css',
  'src/components/UploadZone.tsx',
  'src/components/ResultCard.tsx',
  'src/components/LoadingSpinner.tsx',
  'backend/main.py',
  'backend/ai_processor.py',
  'backend/requirements.txt',
  'README.md'
];

let allFilesExist = true;

console.log('📁 Checking required files:');
requiredFiles.forEach(file => {
  const exists = fs.existsSync(file);
  console.log(`  ${exists ? '✅' : '❌'} ${file}`);
  if (!exists) allFilesExist = false;
});

console.log('\n📦 Checking package.json:');
try {
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  const requiredDeps = ['react', 'react-dom', 'react-dropzone', 'axios', 'lucide-react'];

  requiredDeps.forEach(dep => {
    const exists = packageJson.dependencies && packageJson.dependencies[dep];
    console.log(`  ${exists ? '✅' : '❌'} ${dep}`);
    if (!exists) allFilesExist = false;
  });
} catch (error) {
  console.log('  ❌ Invalid package.json');
  allFilesExist = false;
}

console.log('\n🐍 Checking Python requirements:');
try {
  const requirements = fs.readFileSync('backend/requirements.txt', 'utf8');
  const requiredPkgs = ['fastapi', 'uvicorn', 'opencv-python', 'numpy', 'Pillow'];

  requiredPkgs.forEach(pkg => {
    const exists = requirements.includes(pkg);
    console.log(`  ${exists ? '✅' : '❌'} ${pkg}`);
    if (!exists) allFilesExist = false;
  });
} catch (error) {
  console.log('  ❌ Could not read requirements.txt');
  allFilesExist = false;
}

console.log('\n🎨 Checking Tailwind config:');
try {
  const tailwindConfig = fs.readFileSync('tailwind.config.js', 'utf8');
  const hasContent = tailwindConfig.includes('content') && tailwindConfig.includes('theme');
  console.log(`  ${hasContent ? '✅' : '❌'} Valid Tailwind configuration`);
  if (!hasContent) allFilesExist = false;
} catch (error) {
  console.log('  ❌ Invalid Tailwind config');
  allFilesExist = false;
}

console.log('\n' + '='.repeat(50));

if (allFilesExist) {
  console.log('🎉 All checks passed! Your Baby Face Generator is ready to go!');
  console.log('\n📋 Next steps:');
  console.log('  1. Run: npm install');
  console.log('  2. Run: npm run dev (for frontend)');
  console.log('  3. Run: cd backend && pip install -r requirements.txt && uvicorn main:app --reload');
  console.log('  4. Open http://localhost:3000 in your browser');
} else {
  console.log('❌ Some checks failed. Please review the missing files or configurations.');
}

console.log('\n🚀 Happy coding! 👶');
