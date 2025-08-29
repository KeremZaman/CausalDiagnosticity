TASK=$1
NUM_SAMPLES=$2
MODEL=$3
EDIT_METHOD=$4
EXPL=$5
METRICS=("simulatability" "lanham-truncated" "lanham-mistakes" "lanham-paraphrase" "lanham-filler")
for METRIC in "${METRICS[@]}"
do
    if [ "${EXPL}" == "synthetic" ]; then
        if [ "${EDIT_METHOD}" == "memit" ]; then
            python eval.py --task $TASK --editor-type ${EDIT_METHOD} --model ${MODEL}_edited_${TASK}_v1 --unfaithful-model ${MODEL}_edited_${TASK}_v2 --tokenizer ${MODEL}_edited_${TASK}_v1 --metric $METRIC --num-samples ${NUM_SAMPLES} --results-output results/${MODEL_NAME}/$TASK/${METRIC}_${NUM_SAMPLES}_synth_${EDIT_METHOD}.json --use-synthetic-explanations > logs/${MODEL_NAME}/${TASK}_eval_${METRIC}_${NUM_SAMPLES}_synth_${EDIT_METHOD}.log
        elif [ "${EDIT_METHOD}" == "ice" ]; then
            python eval.py --task $TASK --editor-type ${EDIT_METHOD} --model $MODEL --unfaithful-model $MODEL --tokenizer $MODEL --metric $METRIC --num-samples ${NUM_SAMPLES} --results-output results/${MODEL_NAME}/$TASK/${METRIC}_${NUM_SAMPLES}_synth_${EDIT_METHOD}.json --use-synthetic-explanations > logs/${MODEL_NAME}/${TASK}_eval_${METRIC}_${NUM_SAMPLES}_synth_${EDIT_METHOD}.log
        fi
    elif [ "${EXPL}" == "real" ]; then
        if [ "${EDIT_METHOD}" == "memit" ]; then
            python eval.py --task $TASK --editor-type ${EDIT_METHOD} --model ${MODEL}_edited_${TASK}_v1 --unfaithful-model ${MODEL}_edited_${TASK}_v2 --tokenizer ${MODEL}_edited_${TASK}_v1 --metric $METRIC --num-samples ${NUM_SAMPLES} --results-output results/${MODEL_NAME}/$TASK/${METRIC}_${NUM_SAMPLES}_real_${EDIT_METHOD}.json > logs/${MODEL_NAME}/${TASK}_eval_${METRIC}_${NUM_SAMPLES}_real_${EDIT_METHOD}.log
        elif [ "${EDIT_METHOD}" == "ice" ]; then
            python eval.py --task $TASK --editor-type ${EDIT_METHOD} --model $MODEL --unfaithful-model $MODEL --tokenizer $MODEL --metric $METRIC --num-samples ${NUM_SAMPLES} --results-output results/${MODEL_NAME}/$TASK/${METRIC}_${NUM_SAMPLES}_real_${EDIT_METHOD}.json > logs/${MODEL_NAME}/${TASK}_eval_${METRIC}_${NUM_SAMPLES}_real_${EDIT_METHOD}.log
        fi
    fi
done
