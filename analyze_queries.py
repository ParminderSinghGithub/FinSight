import json

qs = json.load(open('data_btp/processed/evaluation_queries.json'))
text_qs = [q for q in qs if q['modality'] == 'text']
code_qs = [q for q in qs if q['modality'] == 'code']

print('Total queries: ' + str(len(qs)))
print('Text queries with relevant_ids: ' + str(sum(1 for q in text_qs if q.get('relevant_ids'))))
print('Code queries with relevant_ids: ' + str(sum(1 for q in code_qs if q.get('relevant_ids'))))

text_sample = next((q for q in text_qs if q.get('relevant_ids')), None)
code_sample = next((q for q in code_qs if q.get('relevant_ids')), None)

print('\nSample text query with relevant_ids:')
print(text_sample)
print('\nSample code query with relevant_ids:')
print(code_sample)
