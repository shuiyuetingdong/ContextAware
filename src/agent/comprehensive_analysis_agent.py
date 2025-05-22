import dspy
from typing import List, Dict, Tuple, Optional, Literal
from math import ceil
from ..video_dataloader import Video

class comprehensive_analysis_agent_prompt(dspy.Signature):
    """
    综合分析图片是否有害
    这是一个来自于视频评论区中的图片，这个视频与黑人与中国人之间的跨种族关系有关，
    请你结合视频标题及其评论区的整体氛围语境，依据下面给出的分辨harmful或non_harmful图像的原则，
    分析图片当中的元素，判断其是harmful还是non_harmful。请注意，语境仅仅是参考，
    而非绝对准确，请结合principles_for_identifying_harmful_images的内容，做出更加准确的判断。
    """

    video_title: str = dspy.InputField()  # 视频标题
    video_comment_sentiment: str = dspy.InputField(desc='评论区对黑人与中国人之间跨种族关系的整体情感氛围')  # 评论区整体情感
    image: dspy.Image = dspy.InputField()  # 需要分析的图像路径
    comments: str = dspy.InputField(desc='与图片一同出现的评论文本，可选')  # 评论区内容
    principles_for_identifying_harmful_images: str = dspy.InputField()  # 分析图片的原则

    analysis: str = dspy.OutputField() # 分析结果，当只需要分类结果时，可以注释掉这一行
    sentiment: Literal['harmful', 'non_harmful'] = dspy.OutputField(desc='综合判断图片是harmful还是non_harmful') 


class Comprehensive_Analysis_Agent:

    # 分析评论区其余的图片
    @staticmethod
    def run(Video: Video, cover: bool = True) -> List[Optional[Literal["harmful", "non_harmful"]]]:
        """
        分析实例中的图片情感（harmful 或 non_harmful）。
        选取后 90% 的图片（按点赞数排序，向上取整），调用 dspy 分析图片情感。

        :return: 一个列表，包含每张图片的情感分类（harmful 或 non_harmful）。
        """

        if not Video.images:
            print("无图片可分析。")
            return []

        # 按点赞数降序排序
        sorted_images = sorted(Video.images, key=lambda x: x.get("likes", 0), reverse=True)

        # 选取后 90% 的图片（向上取整）
        top_n = ceil(len(sorted_images) * 0.1)
        selected_images = sorted_images[top_n:]

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



            # 调用 dspy 模块进行情感分析
            CA_agent = dspy.Predict(comprehensive_analysis_agent_prompt)
            analysis_result = CA_agent(video_title = Video.title, video_comment_sentiment = Video.overall_sentiment, image = f'../data/image/image_sample/{name}.jpg', comments = text, principles_for_identifying_harmful_images = principles_prompt)


            # 获取图片情感分类
            sentiment = analysis_result.get("sentiment", None)  # 默认值为 None
            image["sentiment"] = sentiment  # 更新到 image 的 sentiment 属性
            analyzed_sentiments.append(sentiment)

        return analyzed_sentiments


# principles for identifying harmful images
principles_prompt = '''1. 语言和符号
检查是否包含侮辱、贬低或攻击性的语言和符号。
注意这些元素是否针对特定种族、文化或跨种族关系。
2. 视觉表达
识别图片中是否存在夸张或丑化特定特征的视觉元素，这些元素可能激发偏见或敌视情感。
3. 语境和意图
结合图片整体的语境分析其意图，判断是否旨在制造分裂、传播仇恨或攻击特定群体。
评估是否通过直接或间接的手段使某些群体感到不被接受、不被尊重或受攻击。
4. 情感激发
判断图片是否渲染负面情绪（如仇恨、不安），以引导观众的情感反应。
5. 种族和文化表现
警惕种族刻板印象，如夸张的面部特征、肤色、服饰等。
观察是否有对文化符号、服饰或习俗的夸张、嘲讽或贬低，这可能导致文化污名化和歧视。
6. 性暗示和性别偏见
注意图片是否包含不当的性暗示，尤其是通过低俗或物化的方式表现跨种族关系。
评估图片是否贬低女性角色或加深性别刻板印象，特别是在跨种族婚姻背景下。
7. 符号性元素和象征意义
识别是否利用负面符号、选择性展示或暗示来贬低某群体的特征或行为。
注意图片是否将特定群体或社会关系与负面事件、性质或道德评价联系起来。
8. 文化和社会影响
判断图片是否通过对文化差异的对比或不当表现贬低跨文化婚姻或领域。
关注其是否助长社会偏见，导致对某些行为、关系或文化的误解。
9. 讽刺与反语
评估图片是否利用反语或讽刺的方式，暗示跨种族关系不正常或不可接受。
10. 直接攻击性
观察图片是否直接包含侮辱性语言、图像或符号，用以贬低、羞辱或攻击某一群体或个体。
11. 污名化元素
注意是否通过视觉或文字传播负面信息，导致对某一群体的不当偏见或刻板印象。
警惕对跨种族关系、性别角色等的污名化表现。
12. 道德争议和社会联系
观察图片是否试图将特定群体或关系与负面的社会观念、道德争议联系起来，从而加深公众中的污名化倾向。'''