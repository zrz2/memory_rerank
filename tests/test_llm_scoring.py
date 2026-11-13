import json
from unittest.mock import Mock
import pytest
from memory_rerank.llm_scoring import BudgetedScorer,messages_for,parse_result

VIEWS=[{'alias':'M001','created_at':'2020-01-01','text':'A personal goal','has_answer':True,'id':'answer_secret'}]
CFG={'max_output_tokens':100,'budget_input_cny_per_million':10,'budget_output_cny_per_million':30,'budget_cny':5,'max_api_calls':2}
def client(text='{"M001":7}'):
    c=Mock(); c.config={'model':'fake'}
    c.chat.return_value={'text':text,'finish_reason':'stop','usage':{'prompt_tokens':50,'completion_tokens':10},'returned_model':'fake'}
    return c

def test_allowlist_and_question_independent_importance():
    m=messages_for('importance',VIEWS,{'text':'SECRET QUESTION','query_time':'today'})
    raw=json.dumps(m)
    assert all(x not in raw for x in ['answer_secret','has_answer','SECRET QUESTION'])
    r=messages_for('rerank',VIEWS,{'text':'question','query_time':'today','answer':'SECRET ANSWER'})
    assert 'SECRET ANSWER' not in json.dumps(r)

@pytest.mark.parametrize('task,raw',[
 ('importance','{"M001":true}'),('importance','{"M001":11}'),('importance','{"M002":5}'),
 ('rerank','["M001","M001"]'),('rerank','["M002"]'),('rerank','[]')])
def test_invalid_output_rejected(task,raw):
    with pytest.raises(ValueError): parse_result(task,raw,['M001'])

def test_cache_reuses_response_without_cost(tmp_path):
    c=client(); s=BudgetedScorer(c,tmp_path,CFG)
    first=s.call('importance',VIEWS)
    second=s.call('importance',VIEWS)
    assert first[0]==second[0]=={'M001':7}
    assert second[2] is True
    assert c.chat.call_count==1 and len(s.ledger['calls'])==1

def test_budget_refuses_before_network(tmp_path):
    c=client();s=BudgetedScorer(c,tmp_path,{**CFG,'budget_cny':0.00001})
    with pytest.raises(RuntimeError,match='budget'): s.call('importance',VIEWS)
    c.chat.assert_not_called()

def test_network_failure_keeps_reservation_and_blocks_retry(tmp_path):
    c=client();c.chat.side_effect=RuntimeError('network')
    s=BudgetedScorer(c,tmp_path,CFG)
    with pytest.raises(RuntimeError,match='network'): s.call('importance',VIEWS)
    s=BudgetedScorer(c,tmp_path,CFG)
    with pytest.raises(RuntimeError,match='Prior request'): s.call('importance',VIEWS)
    assert c.chat.call_count==1 and len(s.ledger['calls'])==1

def test_malformed_reply_saved_without_retry(tmp_path):
    c=client('not JSON');s=BudgetedScorer(c,tmp_path,CFG)
    for _ in range(2):
        with pytest.raises(ValueError):s.call('importance',VIEWS)
    assert c.chat.call_count==1
