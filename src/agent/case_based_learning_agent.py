import dspy
from typing import Literal

class case_based_learning_agent(dspy.Signature):
    """
    提取高赞harmful图片模因
    请结合提供的语境，分析这张有害(harmful)图片的元素，具体说明这张图片在特定语境下是如何有害的
    （例如，使用了怎样的比喻，反语，讽刺，性暗示，污名化，刻板印象或直接攻击等等）。
    """
    
    video_title: str = dspy.InputField()  # 视频标题
    video_comment_sentiment: str = dspy.InputField(desc='评论区对黑人与中国人之间跨种族关系的整体情感氛围')  # 评论区整体情感
    image: dspy.Image = dspy.InputField()  # 需要分析的图像路径

    memes_of_harmful_images: list[str] = dspy.OutputField(desc='请分点分析，并具体说明') 


    # 分析并提取视频评论区当中的高赞harmful图片的模因
    def meme_extractor(self) -> List[str]:
        """
        先选取点赞数排名前 10% 的图片（向上取整），再筛选出 sentiment 为 'harmful' 的图片。
        对这些图片结合 title 和 overall_sentiment 进行模因分析，并将结果返回。

        :return: 一个包含所有图片模因分析结果的列表。
        """
        # 用于存储所有图片的分析结果
        meme_analysis_results = []

        # 按点赞数降序排序，选取前 10% 的图片
        sorted_images = sorted(self.images, key=lambda x: x.get("likes", 0), reverse=True)

        # 选取点赞数前 10% 的图片（向上取整）
        top_n = ceil(len(sorted_images) * 0.1)
        top_images = sorted_images[:top_n]

        # 筛选出 sentiment 为 'harmful' 的图片
        harmful_images = [image for image in top_images if image.get("sentiment") == "harmful"]

        # 如果没有 harmful 的图片，直接返回空结果
        if not harmful_images:
            print("没有 harmful 的图片可以分析。")
            return meme_analysis_results

        # 对每张 harmful 图片进行模因分析
        for image in harmful_images:
            name = image["name"]

            # 调用 dspy 进行模因分析
            ME_agent = dspy.Predict(meme_extraction_agent)
            analysis_result = ME_agent(task = meme_extraction_prompt, video_title = self.title, video_comment_sentiment = self.overall_sentiment, image = f'data/{name}.jpg')

            meme_analysis = analysis_result.get("memes_of_harmful_images", [])

            # 将每个分析结果加入到最终的分析结果列表中
            meme_analysis_results.extend(meme_analysis)

        return meme_analysis_results