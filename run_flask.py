#!/usr/bin/env python3
"""
Flask App Runner for Exam Cheating Detection Dashboard
"""

from app import app

if __name__ == '__main__':
    print("🚀 Starting Exam Cheating Detection Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}") 