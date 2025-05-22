import dspy
from typing import Literal
import random
from typing import Tuple

class comment_processing_agent(dspy.Signature):
    """
    总结视频的评论区整体氛围
    这是一个讲述黑人与中国人之间跨种族关系的视频。请分析视频的标题、封面、评论内容，
    总结这个视频（作者的态度）及其评论区（观众的态度）看待黑人与中国人之间跨种族关系的整体氛围。
    在分析时，应当考虑：
    1.中国对黑人的负面情绪，
    2.使用暗讽和性暗示表达的歧视与污名化内容，
    3.反语，特别是使用重复的表达和简单但是非常正面的词语与表情来表达嘲讽。
    """
    
    video_title: str = dspy.InputField()  # 视频标题
    video_cover: dspy.Image = dspy.InputField()  # 视频封面 URL
    comments: str = dspy.InputField(desc='评论区当中的高赞评论与部分抽取的普通评论，需要结合上下文分析，存在部分无意义内容，可选择性忽略')  # 评论区内容

    # analysis_procedure: dict[str, list[str]] = dspy.OutputField(desc="Analysis of hidden negative emotions") # for CoT
    # video_cover_analysis: str = dspy.OutputField(desc="Analyze the elements in the picture (i.e. the cover of the video)")  # 视频封面分析
    video_comment_sentiment_analysis: str = dspy.OutputField()  # 评论区整体氛围
    sentiment: Literal['positive', 'negative'] = dspy.OutputField(desc='除去难以分析的评论，哪种声音更多')  # 评论区整体情感


    # 分析视频评论区整体氛围
    def comments_analysis(self, cover: bool = True) -> Tuple[Literal["positive", "negative"], str]:
        """
        根据视频标题、封面 URL 和评论内容分析评论区整体氛围。

        :param title: 视频标题
        :param cover_url: 视频封面 URL
        :param comments: 评论列表，每条评论包含文本、点赞数和图片信息。
                         格式为 [{"content": str, "likes": int, "image": Optional[str]}, ...]

        :return: 返回评论区的情感分类以及评论区整体氛围。
        """

        """
        抽取点赞数排名前20的普通评论，再随机抽取其余普通评论中的5条，将这些评论合并为单个字符串并返回。

        合并后的评论字符串，以回车分隔。
        """
        # 如果 cover 为 False 且 overall_sentiment 和 pn_sentiment 已有内容，则跳过
        if not cover and self.overall_sentiment is not None and self.pn_sentiment is not None:
            print(f"视频 {self.video_id} 已有分析结果，跳过分析。")
            return self.pn_sentiment, self.overall_sentiment
        
        # 获取普通评论
        normal_comments = [comment for comment in self.comments if 'content' in comment]

        # 按点赞数降序排序
        sorted_comments = sorted(normal_comments, key=lambda x: x.get('likes', 0), reverse=True)

        # 抽取前20条评论（按点赞数排序）
        top_comments = sorted_comments[:20]

        # 剩余评论
        remaining_comments = sorted_comments[20:]

        # 随机抽取剩余评论中的5条
        random_comments = random.sample(remaining_comments, min(len(remaining_comments), 5))

        # 合并两部分评论
        selected_comments = top_comments + random_comments

        # 将评论内容合并为单个字符串，用回车分隔
        merged_comments = "\n".join(comment['content'] for comment in selected_comments)
        #print(merged_comments)

        CP_agent = dspy.Predict(comment_processing_agent)
        analysis_result = CP_agent(video_title = self.title, video_cover = self.cover_url, comments = top_comments)  # merged_comments

        # CP_agent = dspy.ChainOfThought(comment_processing_agent)
        # analysis_result = CP_agent(task = "这是一个讲述黑人与中国人的视频。请结合视频标题、封面、评论内容，考虑到中国对黑人的微妙情绪，以及评论当中存在的大量暗讽，分析评论区整体氛围", video_title = self.title, comments = merged_comments)

        self.overall_sentiment = analysis_result.video_comment_sentiment_analysis  # 评论区整体情感
        self.pn_sentiment = analysis_result.sentiment  # 评论区整体情感分类, positive or negative

        print(analysis_result)

        return analysis_result.sentiment, analysis_result.video_comment_sentiment_analysis  # 分布返回情感分类和评论区整体氛围
