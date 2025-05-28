set autoscale
plot for [i=1:3] 'x_targets.txt' using 0:(column(2*i-1)+1) with lines lw 2 lc rgb word("red green blue", i), \
                          for [i=1:3] 'x_targets.txt' using 0:(column(2*i)+1) with lines lw 2 dt 2 lc rgb word("red green blue", i), \
                          for [i=1:3] 'y_targets.txt' using 0:(column(2*i-1)+0) with lines lw 2 lc rgb word("red green blue", i), \
                          for [i=1:3] 'y_targets.txt' using 0:(column(2*i)+0) with lines lw 2 dt 2 lc rgb word("red green blue", i)
unset key
replot

set autoscale
plot for [i=0:12] sprintf('ekf_nis_%d.txt', i) using 1 with lines lw 2
unset key
replot
