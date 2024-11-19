# coding=utf-8
"""
    @project: maxkb
    @Author：虎
    @file： base_search_dataset_step.py
    @date：2024/1/10 10:33
    @desc:
"""
import os
import re
import time
from typing import List, Dict

from django.db.models import QuerySet

from application.chat_pipeline.I_base_chat_pipeline import ParagraphPipelineModel
from application.chat_pipeline.step.search_dataset_step.i_search_dataset_step import ISearchDatasetStep
from langchain_core.documents import Document

from common.config.embedding_config import VectorStore, ModelManage
from common.db.search import native_search
from common.util.file_util import get_file_content
from dataset.models import Paragraph, DataSet
from embedding.models import SearchMode
from setting.models import Model
from setting.models_provider import get_model
from smartdoc.conf import PROJECT_DIR


def get_model_by_id(_id, user_id):
    model = QuerySet(Model).filter(id=_id).first()
    if model is None:
        raise Exception("模型不存在")
    if model.permission_type == 'PRIVATE' and str(model.user_id) != str(user_id):
        raise Exception(f"无权限使用此模型:{model.name}")
    return model


def get_embedding_id(dataset_id_list):
    dataset_list = QuerySet(DataSet).filter(id__in=dataset_id_list)
    if len(set([dataset.embedding_mode_id for dataset in dataset_list])) > 1:
        raise Exception("关联知识库的向量模型不一致，无法召回分段。")
    if len(dataset_list) == 0:
        raise Exception("知识库设置错误,请重新设置知识库")
    return dataset_list[0].embedding_mode_id


class BaseSearchDatasetStep(ISearchDatasetStep):

    def execute(self, problem_text: str, dataset_id_list: list[str], exclude_document_id_list: list[str],
                exclude_paragraph_id_list: list[str], top_n: int, similarity: float, padding_problem_text: str = None,
                search_mode: str = None,
                user_id=None,
                **kwargs) -> List[ParagraphPipelineModel]:
        if len(dataset_id_list) == 0:
            return []
        exec_problem_text = padding_problem_text if padding_problem_text is not None else problem_text
        model_id = get_embedding_id(dataset_id_list)
        model = get_model_by_id(model_id, user_id)
        self.context['model_name'] = model.name
        embedding_model = ModelManage.get_model(model_id, lambda _id: get_model(model))
        embedding_value = embedding_model.embed_query(exec_problem_text)
        vector = VectorStore.get_embedding_vector()
        embedding_list = vector.query(exec_problem_text, embedding_value, dataset_id_list, exclude_document_id_list,
                                      exclude_paragraph_id_list, True, top_n, similarity, SearchMode(search_mode))
        if embedding_list is None:
            return []
        paragraph_list = self.list_paragraph(embedding_list, vector)
        
        # paragraph_list=merge_paragraphs(paragraph_list,1)

        rerank_model_id='728583d2-a188-11ef-abd3-26cf8447a8c9'
        
        # reranker_model = get_model_instance_by_model_user_id('rerank_model_id',user_id) 
        # 这样获取模型实例用python main.py start 起服务会直接挂，还没深入研究，感觉不搭嘎。。。
        
        reranker_model = get_model_by_id(rerank_model_id,user_id)
        reranker_model = ModelManage.get_model(rerank_model_id, lambda _id: get_model(reranker_model))
        
        documents=[Document(row.get('content')) for row in paragraph_list]
        
        start_time=time.time()
        
        rerank_results = reranker_model.compress_documents(
            documents,
            exec_problem_text)
        
        
        execution_time = time.time() - start_time
        print(f"reranker_model.compress_documents 执行时间: {execution_time:.4f} 秒")

        rerank_dict = {doc.page_content: doc.metadata['relevance_score'] for doc in rerank_results}
        
        for paragraph in paragraph_list:
            paragraph['similarity'] = rerank_dict[paragraph['content']]
            paragraph['comprehensive_score'] = rerank_dict[paragraph['content']]
        
        result = [self.reset_paragraph(paragraph, embedding_list) for paragraph in paragraph_list if paragraph['similarity'] > similarity]
        
        return result

    @staticmethod
    def reset_paragraph(paragraph: Dict, embedding_list: List) -> ParagraphPipelineModel:
        filter_embedding_list = [embedding for embedding in embedding_list if
                                 str(embedding.get('paragraph_id')) == str(paragraph.get('id'))]
        if filter_embedding_list is not None and len(filter_embedding_list) > 0:
            find_embedding = filter_embedding_list[-1]
            return (ParagraphPipelineModel.builder()
                    .add_paragraph(paragraph)
                    .add_similarity(paragraph.get('similarity'))
                    .add_comprehensive_score(paragraph.get('comprehensive_score'))
                    .add_dataset_name(paragraph.get('dataset_name'))
                    .add_document_name(paragraph.get('document_name'))
                    .add_hit_handling_method(paragraph.get('hit_handling_method'))
                    .add_directly_return_similarity(paragraph.get('directly_return_similarity'))
                    .build())

    @staticmethod
    def get_similarity(paragraph, embedding_list: List):
        filter_embedding_list = [embedding for embedding in embedding_list if
                                 str(embedding.get('paragraph_id')) == str(paragraph.get('id'))]
        if filter_embedding_list is not None and len(filter_embedding_list) > 0:
            find_embedding = filter_embedding_list[-1]
            return find_embedding.get('comprehensive_score')
        return 0

    @staticmethod
    def list_paragraph(embedding_list: List, vector):
        paragraph_id_list = [row.get('paragraph_id') for row in embedding_list]
        if paragraph_id_list is None or len(paragraph_id_list) == 0:
            return []
        paragraph_list = native_search(QuerySet(Paragraph).filter(id__in=paragraph_id_list),
                                       get_file_content(
                                           os.path.join(PROJECT_DIR, "apps", "application", 'sql',
                                                        'list_dataset_paragraph_by_paragraph_id.sql')),
                                       with_table_name=True)                        
        # 如果向量库中存在脏数据 直接删除
        if len(paragraph_list) != len(paragraph_id_list):
            exist_paragraph_list = [row.get('id') for row in paragraph_list]
            for paragraph_id in paragraph_id_list:
                if not exist_paragraph_list.__contains__(paragraph_id):
                    vector.delete_by_paragraph_id(paragraph_id)
        # 如果存在直接返回的则取直接返回段落
        hit_handling_method_paragraph = [paragraph for paragraph in paragraph_list if
                                         (paragraph.get(
                                             'hit_handling_method') == 'directly_return' and BaseSearchDatasetStep.get_similarity(
                                             paragraph, embedding_list) >= paragraph.get(
                                             'directly_return_similarity'))]
        if len(hit_handling_method_paragraph) > 0:
            # 找到评分最高的
            return [sorted(hit_handling_method_paragraph,
                           key=lambda p: BaseSearchDatasetStep.get_similarity(p, embedding_list))[-1]]
        return paragraph_list

    def get_details(self, manage, **kwargs):
        step_args = self.context['step_args']

        return {
            'step_type': 'search_step',
            'paragraph_list': [row.to_dict() for row in self.context['paragraph_list']],
            'run_time': self.context['run_time'],
            'problem_text': step_args.get(
                'padding_problem_text') if 'padding_problem_text' in step_args else step_args.get('problem_text'),
            'model_name': self.context.get('model_name'),
            'message_tokens': 0,
            'answer_tokens': 0,
            'cost': 0
        }
def merge_paragraphs(paragraph_list, num_paragraphs=1):
    processed_paragraphs = set()

    # 遍历每个初始段落，查找并合并相邻段落的内容
    for paragraph in paragraph_list:
        document_sort_id = paragraph.get('document_sort_id')
        if document_sort_id:
            processed_paragraphs.add(document_sort_id)

    for paragraph in paragraph_list:
        document_sort_id = paragraph.get('document_sort_id')
        document_id = paragraph.get('document_id')
        if document_sort_id and document_id:
            match = re.match(r'.*?_?(\d+)$', document_sort_id)
            if match:
                chunk_number = int(match.group(1))

                # 查找前几个段落
                for i in range(1, num_paragraphs + 1):
                    prev_chunk_id = f"{document_id}_{chunk_number - i}"
                    prev_paragraph = Paragraph.objects.filter(document_sort_id=prev_chunk_id, document_id=document_id).first()
                    if prev_paragraph and prev_paragraph.document_sort_id not in processed_paragraphs:
                        paragraph['content'] = f"{prev_paragraph.content}\n{paragraph['content']}"
                        processed_paragraphs.add(prev_paragraph.document_sort_id)

                # 查找后几个段落
                for i in range(1, num_paragraphs + 1):
                    next_chunk_id = f"{document_id}_{chunk_number + i}"
                    next_paragraph = Paragraph.objects.filter(document_sort_id=next_chunk_id, document_id=document_id).first()
                    if next_paragraph and next_paragraph.document_sort_id not in processed_paragraphs:
                        paragraph['content'] += f"\n{next_paragraph.content}"
                        processed_paragraphs.add(next_paragraph.document_sort_id)

    return paragraph_list