for i in `grep -E 'run_start|run_stop' slurm-*.out|egrep -v "Required EXACTLY" |awk -F':' '{print $1}'|sort -nr |uniq`
do
  #echo $i
  total_lines=$(wc -l <$i)
  target_line=$((total_lines - 2))
  run_status=$(sed -n "${target_line}p" $i|awk -F'- ' '{print $2}')
  if [[ "${run_status}" == "SUCCESS" ]]; then
    #echo "Run Successful for ", $i
    start_time_line=$(grep -E 'run_start' $i)
    #start_time_line=$(grep -E 'run_start|run_stop' $i)
    #echo $start_time_line
    start_time=$(echo $start_time_line|sed 's/^.*MLLOG //'|jq '.time_ms')
    #echo $start_time
    stop_time_line=$(grep -E 'run_stop' $i)
    #start_time_line=$(grep -E 'run_start|run_stop' $i)
    #echo $start_time_line
    stop_time=$(echo $stop_time_line|sed 's/^.*MLLOG //'|jq '.time_ms')
    #echo $stop_time
    #duration=$((stop_time / 60000 - $start_time / 60000 ))
    duration=$(echo $stop_time/60000 - $start_time/60000 | bc -l)
    #echo "Time taken is $duration"
    num_nodes=$(grep -E 'num_nodes:' $i|sed 's/^.*num_nodes: //')
    #echo $num_nodes
    logf=$(grep 'INFO - Running compliance on file: ' $i|awk -F'\/' {'print $3'})
    #echo "Log file $logf"
    lr=$(grep 'opt_base_learning_rate' $i|sed 's/^.*MLLOG //'|jq '.value')
    #echo "learning rate $lr"
    printf "%s\t" $i $logf $num_nodes $lr $start_time $stop_time $duration >>out1.tab
    printf '\n' >>out1.tab
  fi
done
