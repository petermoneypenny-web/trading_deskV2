#!/bin/bash

# Target trading days from last week (Mon-Fri)
DATES=("2026-11-02" "2026-11-03" "2026-11-04" "2026-11-05" "2026-11-06")
TOTAL_PNL=0.0

echo "=========================================================="
echo " Starting Weekly Replay with Gemini Flash (Nov 2 - Nov 6) "
echo "=========================================================="
printf "%-12s | %-12s\n" "Date" "Daily P&L"
echo "----------------------------------------------------------"

for DATE in "${DATES[@]}"; do
    # Run the replay test and capture full output
    OUTPUT=$(docker exec trading-desk-v2 python /app/replay_test.py --date "$DATE" --with-llm --max-llm-calls 6)
    
    # Extract the realized P&L value (e.g., +52.50 or -121.93)
    DAY_PNL=$(echo "$OUTPUT" | grep "Final Realized P&L:" | sed -E 's/.*Final Realized P&L: \$\+?(-?[0-9]+\.[0-9]+).*/\1/')

    if [ -z "$DAY_PNL" ]; then
        DAY_PNL="0.00"
    fi

    printf "%-12s | $%s\n" "$DATE" "$DAY_PNL"

    # Accumulate total P&L using awk for floating-point precision
    TOTAL_PNL=$(awk -v t="$TOTAL_PNL" -v d="$DAY_PNL" 'BEGIN { printf "%.2f", t + d }')
done

echo "=========================================================="
printf "TOTAL WEEKLY REALIZED P&L: $%s\n" "$TOTAL_PNL"
echo "=========================================================="
