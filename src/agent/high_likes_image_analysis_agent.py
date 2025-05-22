import dspy
from typing import Literal
from math import ceil
from typing import List, Optional

class high_likes_image_analysis_agent(dspy.Signature):
    """
    分析高赞图片情感
    这是一个来自于视频评论区中的高赞图片，请你结合视频标题及其评论区的整体氛围语境，
    分析这张图片的元素及其可能的含义，并判断其是否是有害的(harmful)。
    请注意，语境仅仅是参考，而非绝对准确，如果你有充足的理由或证据，请做出更加准确的判断。
    """
    
    video_title: str = dspy.InputField()  # 视频标题
    video_comment_sentiment: str = dspy.InputField(desc='评论区对黑人与中国人之间跨种族关系的整体情感氛围')  # 评论区整体情感
    image: dspy.Image = dspy.InputField()  # 需要分析的图像路径
    comments: str = dspy.InputField(desc='与图片一同出现的评论文本，可选')  # 评论区内容

    sentiment: Literal['harmful', 'non_harmful'] = dspy.OutputField(desc='综合判断图片是否是harmful的') 


    # 分析评论区前 10% 的图片
    def image_analysis(self, cover: bool = True) -> List[Optional[Literal["harmful", "non_harmful"]]]:
        """
        分析实例中的图片情感（harmful 或 non_harmful）。
        选取前 10% 的图片（按点赞数排序，向上取整），调用 dspy 分析图片情感。

        :return: 一个列表，包含每张图片的情感分类（harmful 或 non_harmful）。
        """

        if not self.images:
            print("无图片可分析。")
            return []

        # 按点赞数降序排序
        sorted_images = sorted(self.images, key=lambda x: x.get("likes", 0), reverse=True)

        # 选取前 10% 的图片（向上取整）
        top_n = ceil(len(sorted_images) * 0.1)
        selected_images = sorted_images[:top_n]

        # 初始化情感分类列表
        analyzed_sentiments = []
        for image in selected_images:
            # 如果 cover 为 False 且 sentiment 已有内容，则跳过
            if not cover and image.get("sentiment") is not None:
                print(f"图片 {image['name']} 已有情感分析结果，跳过。")
                analyzed_sentiments.append(image["sentiment"])
                continue

            name = image["name"]
            text = image.get("text", "")

            # # 调试功能：打开图片。 注意，不要在大批量运行时使用这个功能。
            # image_path = f"data/{name}.jpg"
            # try:
            #     img = PILImage.open(image_path)
            #     img.show()  # 在默认图片查看器中显示图片
            #     print(f"正在分析图片: {image_path}")
            # except FileNotFoundError:
            #     print(f"图片文件未找到: {image_path}")
            #     analyzed_sentiments.append(None)
            #     continue

            # 调用 dspy 模块进行情感分析
            HLIA_agent = dspy.Predict(high_likes_image_analysis_agent)
            analysis_result = HLIA_agent(video_title = self.title, video_comment_sentiment = self.overall_sentiment, image = f'data/{name}.jpg', comments = text)


            # 获取图片情感分类
            sentiment = analysis_result.get("sentiment", None)  # 默认值为 None
            image["sentiment"] = sentiment  # 更新到 image 的 sentiment 属性
            analyzed_sentiments.append(sentiment)

        return analyzed_sentiments