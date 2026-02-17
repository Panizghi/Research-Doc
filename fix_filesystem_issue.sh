#!/bin/bash

# System-wide File Read Timeout Fix Script
# Run with: bash fix_filesystem_issue.sh

echo "=== System-wide File Read Timeout Diagnostic & Fix ==="
echo ""

# Step 1: Clear system caches (requires sudo)
echo "Step 1: Clearing system caches..."
sudo purge
echo "✓ Cache cleared"
echo ""

# Step 2: Reset Spotlight index (requires sudo)
echo "Step 2: Resetting Spotlight index..."
sudo mdutil -E /
echo "✓ Spotlight index reset"
echo ""

# Step 3: Clear extended attributes from problematic file
echo "Step 3: Clearing extended attributes from problematic file..."
xattr -rc "/Users/paniz/Documents/GitHub/Research-Doc/DONE/abstract_with_figures (1).tex" 2>/dev/null
echo "✓ Extended attributes cleared"
echo ""

# Step 4: Reset user permissions (requires sudo)
echo "Step 4: Resetting user permissions..."
USER_ID=$(id -u)
sudo diskutil resetUserPermissions / $USER_ID
echo "✓ Permissions reset"
echo ""

# Step 5: Verify disk again
echo "Step 5: Verifying disk integrity..."
diskutil verifyVolume / | tail -5
echo ""

# Step 6: Clear system logs cache
echo "Step 6: Clearing system logs..."
sudo log collect --last 1m --output /tmp/system_logs.logarchive 2>/dev/null || echo "Log collection skipped"
echo ""

# Step 7: Restart filesystem services
echo "Step 7: Restarting filesystem services..."
sudo killall -9 diskarbitrationd 2>/dev/null || echo "diskarbitrationd not running"
sudo killall -9 fseventsd 2>/dev/null || echo "fseventsd not running"
echo "✓ Services restarted (will auto-restart)"
echo ""

# Step 8: Test file read
echo "Step 8: Testing file read..."
if timeout 5 cat "/Users/paniz/Documents/GitHub/Research-Doc/DONE/abstract_with_figures (1).tex" > /dev/null 2>&1; then
    echo "✓ File read successful!"
else
    echo "✗ File read still timing out"
    echo ""
    echo "Additional steps you may need:"
    echo "1. Restart your Mac"
    echo "2. Run First Aid in Disk Utility (GUI)"
    echo "3. Check if file is on external/network drive"
    echo "4. Try Safe Mode: Hold Shift during boot"
fi

echo ""
echo "=== Fix script completed ==="
