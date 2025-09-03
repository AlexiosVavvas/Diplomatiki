#!/bin/bash
gnuplot -persist -e "plot 'logs/cbf_log.txt' u 0:1 w l title 'h' lw 2, 'logs/cbf_log.txt' u 0:2 w l title 'PSI' lw 2, 'logs/cbf_log.txt' u 0:3 w l title 'grad_h' lw 2, 'logs/cbf_log.txt' u 0:4 w l title 'U_safe[0]' lw 2, 'logs/cbf_log.txt' u 0:5 w l title 'U_safe[1]' lw 2, 'logs/cbf_log.txt' u 0:6 w l title 'U_safe[2]' lw 2, 'logs/cbf_log.txt' u 0:7 w l title 'U_safe[3]' lw 2; set grid" &

gnuplot -persist -e "plot 'logs/PSI.txt' u 0:1 w l title 'h_ddot' lw 2, 'logs/PSI.txt' u 0:2 w l title 'alpha_1*h_dot' lw 2, 'logs/PSI.txt' u 0:3 w l title 'alpha_2*h' lw 2, 'logs/PSI.txt' u 0:4 w l title 'PSI' lw 2; set grid" &

gnuplot -persist -e "set xrange [0:10]; set yrange [0:10]; set grid; plot 'logs/agent_state.txt' u 2:3 with line lw 2, 'logs/obstacles_points.txt' u 1:2 with points pt 7 lc rgb 'black'" &

gnuplot -persist -e "plot 'logs/agent_state.txt' u 1:4 with line lw 2 title 'U1', 'logs/agent_state.txt' u 1:5 with line lw 2 title 'U2'; set grid" &

gnuplot -persist -e "plot 'logs/ergodic_cost.txt' u 1:2 with line lw 2 title 'Ergodic Cost', 'logs/ergodic_cost.txt' u 1:3 with line lw 2 title 'Active Safe Control'; set grid" &