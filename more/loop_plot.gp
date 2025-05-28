set terminal qt persist
set autoscale
unset key

plot for [i=1:3] 'x_targets.txt' using 0:(column(2*i-1)+1) with lines lw 2 lc rgb word("red green blue", i), \
     for [i=1:3] 'x_targets.txt' using 0:(column(2*i)+1) with lines lw 2 dt 2 lc rgb word("red green blue", i), \
     for [i=1:3] 'y_targets.txt' using 0:(column(2*i-1)) with lines lw 2 lc rgb word("red green blue", i), \
     for [i=1:3] 'y_targets.txt' using 0:(column(2*i)) with lines lw 2 dt 2 lc rgb word("red green blue", i)

print "NIS plot updated"
pause 1
reread
