


import dspy
from PIL import Image as PILImage


from typing import List, Dict, Tuple, Optional, Literal
import pandas as pd
import random
from math import ceil

random.seed(42)




class Video:

    def __init__(
        self,
        video_id: str,
        title: str,
        comments: List[dict],
        images: List[Dict[str, Optional[str | Literal["harmful", "non_harmful"]]]],
        overall_sentiment: Optional[str] = None,
        pn_sentiment: Optional[Literal["positive", "negative"]] = None
    ) -> None:
        """
        初始化单个视频实例。

        :param video_id: 视频 ID
        :param title: 视频标题
        :param comments: 普通评论列表
        :param images: 图片评论列表
        :param overall_sentiment: 评论区整体氛围
        """
        self.video_id = video_id  # 视频 ID
        self.title = title if title else f"无标题_{video_id}"  # 默认标题
        self.comments = comments  # 普通评论
        self.images = images  # 图片评论
        self.overall_sentiment = overall_sentiment  # 评论区整体氛围分析
        self.pn_sentiment = pn_sentiment  # 评论区整体情感分类, positive or negative



    # 从 CSV 文件中解析多个视频并创建对应的 Video 实例
    @staticmethod
    def create_videos_from_csv(csv_path: str) -> Dict[str, 'Video']:
        """
        从 CSV 文件中解析多个视频并创建对应的 Video 实例，使用 video_id 命名实例。

        :param csv_path: 包含多个视频数据的 CSV 文件路径
        :return: 字典，键为 video_'video_id'，值为 Video 实例
        """
        # 加载 CSV 文件
        df = pd.read_csv(csv_path)

        # 确保关键列存在
        required_columns = ['video_id', 'desc', 'comment', 'image_label', 'digg_count']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")

        # 检查是否包含 image_sentiment 列，如没有，添加 image_sentiment 列
        if 'image_sentiment' not in df.columns:
            df['image_sentiment'] = pd.Series(dtype='object')  # 设置为字符串类型，并填充为 None

        # 检查是否包含 pn_sentiment 列，如没有，添加 pn_sentiment 列
        if 'pn_sentiment' not in df.columns:
            df['pn_sentiment'] = pd.Series(dtype='object')  # 明确设置为字符串类型

        # 检查是否包含 overall_sentiment 列，如没有，添加 overall_sentiment 列
        if 'overall_sentiment' not in df.columns:
            df['overall_sentiment'] = pd.Series(dtype='object')  # 明确设置为字符串类型

        # 清理数据
        df['video_id'] = df['video_id'].astype(str)  # 确保 video_id 为字符串

        # 用于存储视频实例
        videos = {}

        # 按 video_id 分组
        grouped = df.groupby('video_id')

        for video_id, group in grouped:
            # 获取视频的标题（标题仅出现在第一条记录中）
            print(video_id)

            # 过滤掉不满足条件的行（允许 desc 为空）
            filtered_group = group.dropna(subset=['video_id'])

            # 检查是否为空
            if filtered_group.empty:
                print(f"Warning: video_id={video_id} has no valid rows after filtering.")
                continue 
            # 获取第一行信息
            video_info_row = filtered_group.iloc[0] 

            video_id = video_info_row['video_id']  # 视频 ID            
            title = video_info_row['desc'] if pd.notna(video_info_row['desc']) else None  # 允许 desc 为空

            # 获取 pn_sentiment
            pn_sentiment = video_info_row['pn_sentiment'] if pd.notna(video_info_row['pn_sentiment']) else None

            # 获取 overall_sentiment
            overall_sentiment = video_info_row['overall_sentiment'] if pd.notna(video_info_row['overall_sentiment']) else None

            comments = []
            images = []

            for _, row in group.iterrows():
                # 判断是否为图片评论
                if pd.notna(row['image_url']):
                    images.append({
                        "name": row['cid'],  # 使用用户cid作为图片名称
                        "likes": int(row['digg_count']) if pd.notna(row['digg_count']) else 0,  # 图片点赞数
                        "text": row['comment'] if pd.notna(row['comment']) else None,  # 图片评论文字
                        "sentiment": row['image_sentiment'] if pd.notna(row['image_sentiment']) else None  # 图片情感
                    })
                # 普通评论
                elif pd.notna(row['comment']):
                    comments.append({
                        "content": row['comment'],  # 评论内容
                        "likes": int(row['digg_count']) if pd.notna(row['digg_count']) else 0  # 评论点赞数
                    })

            # 创建单个视频实例
            instance_name = f"video_{video_id}"
            videos[instance_name] = Video(video_id, title, comments, images, pn_sentiment, overall_sentiment)

        return videos


    # 将 overall_sentiment 和 pn_sentiment 写入 CSV 文件中
    @staticmethod
    def save_comments_sentiment(videos: Dict[str, 'Video'], csv_path: str, output_path: str) -> None:
        """
        将每个视频实例的 overall_sentiment 和 pn_sentiment 写入 CSV 文档中。

        :param videos: 视频实例的字典，键为实例名，值为 Video 实例
        :param csv_path: 原始 CSV 文件路径
        :param output_path: 添加了 overall_sentiment 列的新 CSV 文件路径
        """
        # 读取原始 CSV 文档
        df = pd.read_csv(csv_path)

        # 检查是否包含 video_id 和 Link 列
        required_columns = ['video_id', 'Link']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV 文件中缺少必要列: {col}，无法关联视频实例。")        

        # 检查是否包含 overall_sentiment 列，如没有，添加 overall_sentiment 列
        if 'overall_sentiment' not in df.columns:
            df['overall_sentiment'] = pd.Series(dtype='object')  # 明确设置为字符串类型

        # 检查是否包含 pn_sentiment 列，如没有，添加 pn_sentiment 列
        if 'pn_sentiment' not in df.columns:
            df['pn_sentiment'] = pd.Series(dtype='object')  # 明确设置为字符串类型

        # 遍历视频实例，将 sentiment 写入存有标题信息（Link 列有内容）的行
        for video in videos.values():

            # 确定符合条件的行：video_id 匹配，且 Link 列不为空
            condition = (df['video_id'].astype(str) == video.video_id) & df['Link'].notna()

            # 更新 overall_sentiment
            df.loc[condition, 'overall_sentiment'] = video.overall_sentiment
            # 更新 pn_sentiment
            df.loc[condition, 'pn_sentiment'] = video.pn_sentiment


        # 保存到新的 CSV 文件
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"已保存到 {output_path}")


    # 将图片情感 (sentiment) 写入 CSV 文件中
    @staticmethod
    def save_images_sentiment(videos: Dict[str, 'Video'], input_csv: str, output_csv: str) -> None:
        """
        将视频实例的图片情感 (sentiment) 存入 CSV 文件。
        对于每张图片，将 sentiment 写入对应行的 image_sentiment 列。

        :param videos: 包含视频实例的字典，键为实例名，值为 Video 实例。
        :param input_csv: 原始 CSV 文件路径。
        :param output_csv: 输出的 CSV 文件路径。
        """
        # 读取原始 CSV 文件
        df = pd.read_csv(input_csv)

        # 确保包含 image_sentiment 列，如果不存在则创建
        if 'image_sentiment' not in df.columns:
            df['image_sentiment'] = pd.Series(dtype='object')  # 设置为字符串类型

        # 遍历视频实例，将图片的 sentiment 写入对应行
        for video in videos.values():
            for image in video.images:
                # 匹配视频 ID 和图片名称，找到对应的行
                # 此处需要注意video_id, cid, name的类型。如果类型不匹配，sentiment将无法存入。
                condition = (
                    (df['video_id'].astype(str) == video.video_id) & 
                    (df['cid'].astype(int) == image['name']) # 图片名称。在video class中，图片名称使用cid作为名称。在当前文件和打开方式下，name为int类型
                )

                # 更新 image_sentiment 列
                df.loc[condition, 'image_sentiment'] = image.get('sentiment', None)

        # 保存结果到新的 CSV 文件
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"图片情感数据已保存到 {output_csv}")











