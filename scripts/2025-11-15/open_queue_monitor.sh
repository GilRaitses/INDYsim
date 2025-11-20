#!/bin/bash
# Launch Queue Monitor for INDYsim Analysis Queue
# This monitor displays the status of all analyses in the queue,
# including track-by-track progress and console output.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default refresh interval (seconds)
REFRESH_INTERVAL="${1:-0.5}"

# Detect OS and open appropriate terminal
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use Terminal.app or iTerm2
    if command -v iterm2 &> /dev/null || [ -d "/Applications/iTerm.app" ]; then
        # Use iTerm2 if available
        osascript <<EOF
tell application "iTerm"
    activate
    create window with default profile
    tell current session of current window
        write text "cd '$PROJECT_ROOT' && python3 '$SCRIPT_DIR/queue_monitor.py' --refresh-interval $REFRESH_INTERVAL"
    end tell
end tell
EOF
    else
        # Use Terminal.app
        osascript <<EOF
tell application "Terminal"
    activate
    do script "cd '$PROJECT_ROOT' && python3 '$SCRIPT_DIR/queue_monitor.py' --refresh-interval $REFRESH_INTERVAL"
end tell
EOF
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - try common terminals
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "cd '$PROJECT_ROOT' && python3 '$SCRIPT_DIR/queue_monitor.py' --refresh-interval $REFRESH_INTERVAL; exec bash"
    elif command -v xterm &> /dev/null; then
        xterm -e "cd '$PROJECT_ROOT' && python3 '$SCRIPT_DIR/queue_monitor.py' --refresh-interval $REFRESH_INTERVAL"
    else
        echo "ERROR: No suitable terminal found. Please run manually:"
        echo "  python3 $SCRIPT_DIR/queue_monitor.py --refresh-interval $REFRESH_INTERVAL"
        exit 1
    fi
else
    echo "ERROR: Unsupported OS: $OSTYPE"
    exit 1
fi

