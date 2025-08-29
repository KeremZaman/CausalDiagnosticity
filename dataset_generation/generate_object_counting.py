from SPARQLWrapper import SPARQLWrapper, JSON
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError
from transformers import set_seed
import random
import json
from typing import Dict
from itertools import chain, groupby
import os
import argparse
import time
from tqdm import tqdm

RELATION_DATA = {
    'object': {
        'animal': {}, 'musical instrument': {}, 'fruit': {}, 'vegetable': {}, 'furniture': {},
    },
    'occupation': {
        'scientist': {}, 'politician': {}, 'soccer player': {}, 'actor': {}, 'singer': {}
    },
    'company': {
        'media company': {}, 'energy company': {}, 'software company': {}, 'automotive company': {}, 'consulting company': {}
    },
    'touristic attraction': {
        'France': {}, 'Spain': {}, 'Russia': {}, 'Turkey': {}, 'Italy': {}
    },
    'abstract': { 
        'religion': {}, 'political ideology': {}, 'language': {}, 'branch of science': {}, 'emotion': {}
    }
}

#RELATION_DATA = {
#    'object': {
#        'animal': {}, 'musical instrument': {}, 'fruit': {}, 'vegetable': {}, 'furniture': {},
#    }
#    }


REL_VERBALIZER = {'object': 'is ', 'occupation': 'is ', 'company': 'is ', 'touristic attraction': 'is located in ', 'abstract': 'is '}
TYPE_TO_WIKIDATA = {
    'animal': 'Q729', 'musical instrument': 'Q34379', 'fruit': 'Q3314483', 'vegetable': 'Q11004', 'furniture': 'Q14745',
    'scientist': 'Q901', 'politician': 'Q82955', 'soccer player': 'Q937857', 'actor': 'Q33999', 'singer': 'Q177220',
    'media company': 'Q1331793', 'energy company': 'Q1341478', 'software company': 'Q1058914', 'automotive company': 'Q786820', 'consulting company': 'Q2089936',
    'religion': 'Q9174', 'political ideology': 'Q12909644', 'language': 'Q34770', 'branch of science': 'Q2465832', 'emotion': 'Q9415',
    'France': 'Q142', 'Spain': 'Q29', 'Italy': 'Q38', 'Russia': 'Q159', 'Turkey': 'Q43'
    }
REL_TO_WIKIDATA = {'object': 'P279', 'occupation': 'P106', 'company': 'P31', 'touristic attraction': 'P17', 'abstract': 'P31'}

CHANGE_TRACK = {}


def fill_relation_data(dataset_path):
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            return data
    else:
        data = RELATION_DATA
    
    sparqlwd = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = """SELECT ?item ?itemLabel WHERE {{
        ?item wdt:{relation_id} wd:{type_id}.
        {special_clause}
        ?item wikibase:sitelinks ?sitelinks.
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY DESC(?sitelinks)
        LIMIT 20"""
    
    faster_query = """SELECT ?item ?itemLabel WHERE {{
    	SELECT ?item ?itemLabel WHERE {{
        		?item wdt:{relation_id} wd:{type_id}.
        		{special_clause}
        		?item wikibase:sitelinks ?sitelinks.
        		SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        		}}
        		LIMIT 10000
        }}
        ORDER BY DESC(?sitelinks)
        LIMIT 20"""
    
    for relation in tqdm(data):
        rel_id = REL_TO_WIKIDATA[relation]
        for entity_type in tqdm(data[relation]):
            type_id = TYPE_TO_WIKIDATA[entity_type]
            special_clause = '?item wdt:P31 wd:Q570116.' if relation == 'touristic attraction' else ''
            sparqlwd.setQuery(query.format(relation_id=rel_id, type_id=type_id, special_clause=special_clause))
            sparqlwd.setReturnFormat(JSON)
            # if any network issue occured, wait and repeat until getting results
            while True:
                try:
                    results = sparqlwd.query().convert()['results']['bindings']
                except:
                    print(f"An error occured while fetching results for {relation}/{entity_type}. Trying again..")
                    time.sleep(5)
                    sparqlwd.setQuery(faster_query.format(relation_id=rel_id, type_id=type_id, special_clause=special_clause))
                    continue
                break
            entities = [res['itemLabel']['value'] for res in results]
            entities = random.sample(entities, 10)
            data[relation][entity_type] = {'not_changed': entities, 'changed': []}
    
    with open(dataset_path, 'w') as f:
            json.dump(data, f, indent=4)

    return data


def generate_edits(data: Dict, edit_out: str, relation_out: str):
    if os.path.exists(edit_out) and os.path.exists(relation_out):
        with open(edit_out, 'r') as f:
            editing_dataset = json.load(f)

        with open(relation_out, 'r') as f:
            data = json.load(f)

        for item in editing_dataset:
            sample, new_type = item['subject'], item['target_2']
            CHANGE_TRACK[sample] = new_type

        return data

    editing_dataset = []
    # change type of 20% of entities for each category
    for relation in data:
        entity_types = list(data[relation].keys())
        for entity_type in entity_types:
            if len(data[relation][entity_type]['changed']) == 0:
                original_types = data[relation][entity_type]['not_changed']
                
                k = int(0.2 * len(original_types))
                samples = random.sample(original_types, k)
                
                for sample in samples:
                    original_types.remove(sample)
                    new_type = random.choice([t for t in entity_types if t != entity_type])
                    data[relation][entity_type]['changed'].append(sample)
                    CHANGE_TRACK[sample] = new_type

                    subject, relation_txt, old_target, new_target = sample, REL_VERBALIZER[relation], entity_type, new_type
                    editing_dataset.append({'subject': subject, 'true_target': old_target,
                     'target_1': old_target, 'target_2': new_target,
                      'prompt': f'{subject} {relation_txt}'})

    with open(edit_out, 'w') as f:
        json.dump(editing_dataset, f, indent=4)

    with open(relation_out, 'w') as f:
        json.dump(data, f, indent=4)

    return data

def _generate_synth_expl(entities, relation, type_in_question):
    #print(f"entities: {entities} relation: {relation} tiq: {type_in_question}")
    return f"{', '.join(entities)} {'is' if len(entities) == 1 else 'are'} {'located in ' if relation == 'touristic attraction' else ''}{type_in_question}."

def _generate_yes_no_question(data: Dict, relation: str):
    entity_types = list(data.keys())
    entity_to_type = {e: t for t in data for e in chain(*data[t].values())}
    # entity type to be asked about in question
    type_in_question = random.choice(entity_types)
    # decide how many entities will exist in question
    num_entities = random.randint(3, 6)
    # select k entities of type X all of which will be in to-be-changed list
    # at least one slot allocated for other entity types
    k = random.randint(1, min(num_entities - 1, len(data[type_in_question]['changed'])))
    selected_type_entities = random.sample(data[type_in_question]['changed'], k)

    rem = num_entities - k

    changed_to_curr_type = [entity for entity, new_type in CHANGE_TRACK.items() if new_type == type_in_question]
    if len(changed_to_curr_type) == 0:
        return None

    k = random.randint(1, min(rem, len(changed_to_curr_type)))
    other_type_entities = random.sample(changed_to_curr_type, k)
    rem -= k

    all_entities = selected_type_entities + other_type_entities

    # entities from selected type but not changed after edit
    all_selected_type_entities_constant = list(chain(*[data[type]['not_changed'] for type in entity_types if type == type_in_question]))
    all_entities.extend(random.sample(all_selected_type_entities_constant, min(rem, len(all_selected_type_entities_constant))))
    random.shuffle(all_entities)

    # might need to create question using mistral
    choices = ['yes', 'no']
    random.shuffle(choices)
    if random.random() < 0.5:
        question = f"Are any of them {'located in ' if relation == 'touristic attraction' else ''}{type_in_question}? {', '.join(all_entities)}."
        gold_answer = 'yes'
        selected_type_entities_before_edit = list(filter(lambda x: x not in other_type_entities, all_entities))
        selected_type_entities_after_edit = list(filter(lambda x: x not in selected_type_entities, all_entities))
        synth_expl_1 = _generate_synth_expl(selected_type_entities_before_edit, relation, type_in_question)
        synth_expl_2 = _generate_synth_expl(selected_type_entities_after_edit, relation, type_in_question)
    else:
        question = f"Are all of them {'located in ' if relation == 'touristic attraction' else ''}{type_in_question}? {', '.join(all_entities)}."
        gold_answer = 'no'
        # all entities not belonging to questioned type
        other_entities_before_edit = other_type_entities
        other_entities_after_edit = selected_type_entities
        # group other entities wrt their types
        #group_entities = lambda entity_list: [(t, list(g)) for t, g in groupby(entity_list, lambda x: entity_to_type[x])]
        
        groups_before_edit = [(t, list(g)) for t, g in groupby(other_entities_before_edit, lambda x: entity_to_type[x])]
        groups_after_edit = [(t, list(g)) for t, g in groupby(other_entities_after_edit, lambda x: CHANGE_TRACK[x])]
        # get statement explaining which type given group belong to
        be_synth_expls = [_generate_synth_expl(group, relation, type) for type, group in groups_before_edit]
        ae_synth_expls = [_generate_synth_expl(group, relation, type) for type, group in groups_after_edit]
        # combine them
        synth_expl_1 = '; '.join(map(lambda x: x.strip('.'), be_synth_expls)) + '.'
        synth_expl_2 = '; '.join(map(lambda x: x.strip('.'), ae_synth_expls)) + '.'

    related_edits = []
    for entity in selected_type_entities:
        subject, relation_txt, old_target, new_target = entity, REL_VERBALIZER[relation], type_in_question, CHANGE_TRACK[entity]
        #print(f"rel verbalizer: {REL_VERBALIZER} relation: {relation} relation_txt: {relation_txt}")
        related_edits.append({'subject': subject, 'true_target': old_target,
                                'target_1': old_target, 'target_2': new_target,
                                'prompt': f'{subject} {relation_txt}'})
    for entity in other_type_entities:
        #print(entity_to_type)
        subject, relation_txt, old_target, new_target = entity, REL_VERBALIZER[relation], entity_to_type[entity], type_in_question
        related_edits.append({'subject': subject, 'true_target': old_target,
                                'target_1': old_target, 'target_2': new_target,
                                'prompt': f'{subject} {relation_txt}'})

    return {"question": question, "entity_list": all_entities, "choice_A": choices[0],
            "choice_B": choices[1], "label_txt": gold_answer, "label": ['A', 'B'][choices.index(gold_answer)],
            "questioned_entity_type": type_in_question, "main_entities_to_change": selected_type_entities,
            "other_entities_to_change": other_type_entities, "question_type": "yes_no_question",
            "synthetic_explanation_1": synth_expl_1, "synthetic_explanation_2": synth_expl_2,
            "related_edits": related_edits}


def _generate_number_question(data: Dict, relation: str):
    entity_types = list(data.keys())
    entity_to_type = {e: t for t in data for e in chain(*data[t].values())}
    # entity type to be asked about in question
    type_in_question = random.choice(entity_types)
    # decide how many entities will exist in question
    num_entities = random.randint(3, 6)
    # select k entities of type X one of which will be in to-be-changed list
    # at least one slot allocated for other entity types
    k = random.randint(1, num_entities-2)
    selected_entities = random.sample(data[type_in_question]['not_changed'], k)
    selected_entity_to_change = random.choice(data[type_in_question]['changed'])
    selected_entities.append(selected_entity_to_change)
    
    # select remaining entities from other types
    rem = num_entities - len(selected_entities)
    
    other_entities_all = list(chain(*[data[type]['not_changed'] for type in entity_types if type != type_in_question]))
    other_entities = random.sample(other_entities_all, rem - 1)

    # one of them should be in the list to-be-changed to type X, so in alternative question number of type X entities
    # will be kept constant
    potential_changed = [entity for entity, new_type in CHANGE_TRACK.items() if new_type == type_in_question]
    if len(potential_changed) == 0:
        return None
    other_entity_to_change = random.choice(potential_changed)
    other_entities.append(other_entity_to_change)

    # generate question
    all_entities = other_entities + selected_entities
    random.shuffle(all_entities)
    # might need to create question using mistral
    question = f"How many of them are {'located in ' if relation == 'touristic attraction' else ''}{type_in_question}? {', '.join(all_entities)}."

    # generate multiple choices
    gold_answer = len(selected_entities)
    other_choice = len(selected_entities)
    while other_choice == gold_answer:
        other_choice = random.randint(min(1, gold_answer-3), max(gold_answer+3, len(all_entities)))
    
    choices = [str(gold_answer), str(other_choice)]
    gold_answer = str(gold_answer)
    random.shuffle(choices)
    #question += f"Choices: (A) {choices[0]} (B) {choices[1]}"

    synth_expl_1 = _generate_synth_expl(selected_entities, relation, type_in_question)
    counterfactual_entites = [other_entity_to_change if e == selected_entity_to_change else e for e in selected_entities]
    synth_expl_2 = _generate_synth_expl(counterfactual_entites, relation, type_in_question)

    related_edits = []
    subject, relation_txt, old_target, new_target = selected_entity_to_change, REL_VERBALIZER[relation], type_in_question, CHANGE_TRACK[selected_entity_to_change]
    related_edits.append({'subject': subject, 'true_target': old_target,
                          'target_1': old_target, 'target_2': new_target,
                          'prompt': f'{subject} {relation_txt}'})
    subject, relation_txt, old_target, new_target = other_entity_to_change, REL_VERBALIZER[relation], entity_to_type[other_entity_to_change], type_in_question
    related_edits.append({'subject': subject, 'true_target': old_target,
                          'target_1': old_target, 'target_2': new_target,
                          'prompt': f'{subject} {relation_txt}'})

    return {"question": question, "entity_list": all_entities, "choice_A": choices[0],
            "choice_B": choices[1], "label_txt": gold_answer, "label": ['A', 'B'][choices.index(gold_answer)],
            "questioned_entity_type": type_in_question, "main_entities_to_change": [selected_entity_to_change],
            "other_entities_to_change": [other_entity_to_change], "question_type": "number_question",
            "synthetic_explanation_1": synth_expl_1, "synthetic_explanation_2": synth_expl_2,
            "related_edits": related_edits}


def _generate_question_id(item):
    q_type = item['question_type']
    tiq = f"T-{item['questioned_entity_type']}"
    to_change = f"C-{' '.join(item['main_entities_to_change'])}"
    other_to_change = f"OC-{' '.join(item['other_entities_to_change'])}"
    others = f"O-{' '.join(sorted(item['entity_list']))}"
    question_id = f"{q_type}_{tiq}_{to_change}_{other_to_change}_{others}"

    return question_id


def verify_item(item, data):
    """"questioned_entity_type": type_in_question, "main_entities_to_change": selected_entities,
            "other_entities_to_change": other_entities, "question_type": "any_question"}"""
    entity_to_type = {entity: entity_type for rel in data for entity_type in data[rel] for entity in chain(*data[rel][entity_type].values())}
    entity_list = item['entity_list']
    tiq = item['questioned_entity_type']
    tiq_list = [entity for entity in item['entity_list'] if entity_to_type[entity] == tiq]
    others_list = [entity for entity in item['entity_list'] if entity_to_type[entity] != tiq]

    changing_from_tiq = [entity for entity in tiq_list if entity in CHANGE_TRACK and CHANGE_TRACK[entity] != tiq]
    changing_to_tiq = [entity for entity in others_list if entity in CHANGE_TRACK and CHANGE_TRACK[entity] == tiq]

    num_changing_from_tiq, num_changing_to_tiq = len(changing_from_tiq), len(changing_to_tiq)

    if item['question_type'] == 'number_question':

        assert num_changing_from_tiq > 0, "No questioned type being edited"
        assert num_changing_to_tiq > 0, "No other type being edited"
        assert num_changing_from_tiq == num_changing_to_tiq, "Number of entities belonging to questioned type is not the same before and after edit"

    elif item['question_type'] == 'yes_no_question':
        tiq_list_after_edit = list(set(tiq_list).difference(set(changing_from_tiq))) + changing_to_tiq
        if item['label_txt'] == 'yes': # any-typed question
            assert len(set(changing_from_tiq).intersection(changing_to_tiq)) == 0, "Conflicting entities"
            assert len(tiq_list) > 0, "No item of questioned type before edit"
            assert len(tiq_list_after_edit) > 0, "No item of questioned type after edit"
        else: # all-typed question
            assert len(tiq_list) < len(entity_list), "All items are only of questioned type"
            assert len(tiq_list_after_edit) < len(entity_list), "All items are only of questioned type after edit"
            assert len(tiq_list_after_edit) > 0, "No item of questioned type after edit"


def generate_questions(func, data, max_samples_per_relation: int, min_samples_per_relation: int = 10,
                              max_trial_per_relation: int = 1000):
    dataset = []
    question_tracker = []
    for relation in data:
        num_samples_generated = 0
        num_trials = 0

        while num_samples_generated < max_samples_per_relation:
            if num_trials >= max_trial_per_relation and num_samples_generated >= min_samples_per_relation:
                break

            item = func(data[relation], relation)
            num_trials += 1

            if item is None:
                continue

            question_id = _generate_question_id(item)
            if question_id in question_tracker:
                continue

            try:
                verify_item(item, data)
            except AssertionError as e:
                print(e)
                print(f"Ignoring generated item because not verified. Item: {item} ")
                continue

            question_tracker.append(question_id)
            dataset.append(item)
            num_samples_generated += 1

    return dataset


def generate_dataset(proto_relation_dataset_path, editing_dataset_path: str, relation_dataset_path: str, oc_dataset_path: str,
                     max_samples_per_relation: int, min_samples_per_relation):

    all_data = fill_relation_data(proto_relation_dataset_path)
    all_data = generate_edits(all_data, editing_dataset_path, relation_dataset_path)

    dataset = []
    dataset.extend(generate_questions(_generate_number_question, all_data, max_samples_per_relation, min_samples_per_relation))
    dataset.extend(generate_questions(_generate_yes_no_question, all_data, max_samples_per_relation, min_samples_per_relation))

    with open(oc_dataset_path, 'w') as f:
        json.dump(dataset, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate object counting dataseet")
    parser.add_argument("--proto-relation-dataset", type=str, help="Path to the draft of relation dataset")
    parser.add_argument("--editing-dataset", type=str, help="Path to the editing dataset")
    parser.add_argument("--relation-dataset", type=str, help="Path to the relation dataset")
    parser.add_argument("--dataset", type=str, help="Path to the object counting dataset")
    parser.add_argument("--max-samples-per-relation", type=int, help="Maximum number of samples for each question type")
    parser.add_argument("--min-samples-per-relation", type=int, help="Minimum number of samples for each question type")
    parser.add_argument('--seed', type=int, required=False, default=123)

    args = parser.parse_args()
    set_seed(args.seed)

    generate_dataset(args.proto_relation_dataset, args.editing_dataset, args.relation_dataset, args.dataset,
                     args.max_samples_per_relation, args.min_samples_per_relation)
