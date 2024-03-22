id=$1
jizhi=$2
cuda_id=0
group="$(date '+%Y%m%d')$id"
group="20240322$id"


if [ -z "$jizhi" ]; then
    logdir="~/results/rdt"
    dataset_path="/apdcephfs/share_1563664/ztjiaweixu/datasets/RDT"
elif [ "$jizhi" = "sz" ]; then
    logdir="/apdcephfs/share_1563664/ztjiaweixu/rdt_sz"
    dataset_path="/apdcephfs/share_1563664/ztjiaweixu/datasets/RDT"
elif [ "$jizhi" = "cq" ]; then
    logdir="/apdcephfs/share_1150325/ztjiaweixu/rdt_cq"
    dataset_path="/apdcephfs/share_1150325/ztjiaweixu/datasets/RDT"
fi

corruption_agent=EDAC
corruption_seed=2023
corruption_mode="none"  # none, random, adversarial
corruption_obs=0.0
corruption_act=0.0
corruption_rew=0.0
corruption_next_obs=0.0
corruption_rate=0.0
# for env in hopper
# for env in walker2d
for env in halfcheetah
do
    seed=0
    for i in $(seq 4)
    do
        export CUDA_VISIBLE_DEVICES=$cuda_id
        if [ "$corruption_mode" = "random" ]; then
            corruption_seed=$seed
        fi

        tag=$(date "+%Y%m%d%H%M%S")
        python experiment-d4rl/experiment.py --seed ${seed} --env ${env} \
        --corruption_agent ${corruption_agent} \
        --corruption_seed ${corruption_seed} \
        --corruption_mode ${corruption_mode} \
        --corruption_obs ${corruption_obs} \
        --corruption_act ${corruption_act} \
        --corruption_rew ${corruption_rew} \
        --corruption_next_obs ${corruption_next_obs} \
        --corruption_rate ${corruption_rate} \
        --dataset_path ${dataset_path} --group ${group} --outdir ${logdir} \
        > ~/logs/${env}_${seed}_${tag}.out 2> ~/logs/${env}_${seed}_${tag}.err &
        echo "run $cuda_id $env $seed $corruption_seed $tag"
        sleep 2.0
        let seed=$seed+1
        let cuda_id=$cuda_id+1
    done
done

if [ -n "$jizhi" ]; then
    python taiji/run_gpu.py
fi

# ps -ef | grep experiment | awk '{print $2}'| xargs kill -9
