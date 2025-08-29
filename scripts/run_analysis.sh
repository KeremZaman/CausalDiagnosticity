TASK=$1
NUM_SAMPLES=$2
MODEL=$3
EDIT_METHOD=$4
EXPL=$5
METRICS=("lanham-filler-continuous-nr" "lanham-filler-continuous-star" "lanham-filler-continuous-star-nr" "lanham-filler-continuous-dash" "lanham-filler-continuous-dash-nr" "lanham-filler-continuous-dollar" "lanham-filler-continuous-dollar-nr" "lanham-filler-continuous-pilcrow" "lanham-filler-continuous-pilcrow-nr" "lanham-truncated-continuous-informed")

for METRIC in "${METRICS[@]}"
do
    if [ "${EXPL}" == "synthetic" ]; then
        if [ "${EDIT_METHOD}" == "memit" ]; then
            python eval.py --task $TASK --editor-type ${EDIT_METHOD} --model ${EDITED_MODELS}/$TASK/${MODEL}_edited_${TASK}_v1 --unfaithful-model ${EDITED_MODELS}/$TASK/${MODEL}_edited_${TASK}_v2 --tokenizer $MODEL --metric $METRIC --num-samples ${NUM_SAMPLES} --results-output results/${MODEL_NAME}/$TASK/${METRIC}_${NUM_SAMPLES}_synth_${EDIT_METHOD}.json --use-synthetic-explanations > logs/${MODEL_NAME}/${TASK}_eval_${METRIC}_${NUM_SAMPLES}_synth_${EDIT_METHOD}.log
        elif [ "${EDIT_METHOD}" == "ice" ]; then
            python eval.py --task $TASK --editor-type ${EDIT_METHOD} --model ${PRETRAINED_MODELS}/$MODEL --unfaithful-model ${PRETRAINED_MODELS}/$MODEL --tokenizer $MODEL --metric $METRIC --num-samples ${NUM_SAMPLES} --results-output results/${MODEL_NAME}/$TASK/${METRIC}_${NUM_SAMPLES}_synth_${EDIT_METHOD}.json --use-synthetic-explanations > logs/${MODEL_NAME}/${TASK}_eval_${METRIC}_${NUM_SAMPLES}_synth_${EDIT_METHOD}.log
        fi
    elif [ "${EXPL}" == "real" ]; then
        if [ "${EDIT_METHOD}" == "memit" ]; then
            python eval.py --task $TASK --editor-type ${EDIT_METHOD} --model ${EDITED_MODELS}/$TASK/${MODEL}_edited_${TASK}_v1 --unfaithful-model ${EDITED_MODELS}/$TASK/${MODEL}_edited_${TASK}_v2 --tokenizer $MODEL --metric $METRIC --num-samples ${NUM_SAMPLES} --results-output results/${MODEL_NAME}/$TASK/${METRIC}_${NUM_SAMPLES}_real_${EDIT_METHOD}.json > logs/${MODEL_NAME}/${TASK}_eval_${METRIC}_${NUM_SAMPLES}_real_${EDIT_METHOD}.log
        elif [ "${EDIT_METHOD}" == "ice" ]; then
            python eval.py --task $TASK --editor-type ${EDIT_METHOD} --model ${PRETRAINED_MODELS}/$MODEL --unfaithful-model ${PRETRAINED_MODELS}/$MODEL --tokenizer $MODEL --metric $METRIC --num-samples ${NUM_SAMPLES} --results-output results/${MODEL_NAME}/$TASK/${METRIC}_${NUM_SAMPLES}_real_${EDIT_METHOD}.json > logs/${MODEL_NAME}/${TASK}_eval_${METRIC}_${NUM_SAMPLES}_real_${EDIT_METHOD}.log
        fi
    fi
done
