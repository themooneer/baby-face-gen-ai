#!/bin/bash

echo "📊 Baby Face Generator - Log Viewer"
echo "===================================="

# Check if backend.log exists
if [ -f "backend/backend.log" ]; then
    echo "📁 Found backend.log file"
    echo ""
    echo "Choose an option:"
    echo "1. View real-time logs (tail -f)"
    echo "2. View last 50 lines"
    echo "3. View all logs"
    echo "4. Search for errors"
    echo "5. View logs by date"
    echo ""
    read -p "Enter your choice (1-5): " choice

    case $choice in
        1)
            echo "🔄 Viewing real-time logs (Press Ctrl+C to stop)..."
            tail -f backend/backend.log
            ;;
        2)
            echo "📄 Last 50 lines:"
            tail -50 backend/backend.log
            ;;
        3)
            echo "📄 All logs:"
            cat backend/backend.log
            ;;
        4)
            echo "🔍 Searching for errors..."
            grep -i "error\|exception\|failed" backend/backend.log
            ;;
        5)
            echo "📅 Logs by date (last 24 hours):"
            grep "$(date '+%Y-%m-%d')" backend/backend.log
            ;;
        *)
            echo "❌ Invalid choice"
            ;;
    esac
else
    echo "❌ No backend.log file found"
    echo ""
    echo "To create logs, start the backend:"
    echo "cd backend"
    echo "source venv/bin/activate"
    echo "uvicorn main:app --reload --host 0.0.0.0 --port 8000"
    echo ""
    echo "Or run in background:"
    echo "cd backend && source venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000 > backend.log 2>&1 &"
fi

