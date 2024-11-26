import logging
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

max_kb_error = logging.getLogger(__file__)
max_kb = logging.getLogger("max_kb")


class LiveStreamScriptGenerator:
    def __init__(self, chat_model):
        self.llm = chat_model
        self.dong_style = """
        亲爱的朋友们，大家好。在这个信息爆炸的时代，我们每天都在被各种声音和图像所包围，有时候，我们渴望的不过是一片宁静，一段属于自己的时光。今天，我要与大家分享的，不仅仅是一件商品，而是一种生活的艺术，一段时光的回响。想象一下，在一个慵懒的午后，阳光透过窗帘的缝隙，洒在了你手中的这本书上。这不是一本普通的书，它是知识的海洋，是智慧的灯塔。每一页，都承载着作者的心血，每一行，都闪烁着思想的光芒。它不单是文字的堆砌，它是灵魂的对话，是心灵的触碰。今天，我要带来的，就是这样一本书。它不仅仅是阅读的工具，它是你与世界沟通的桥梁。在忙碌的生活中，它能让你找到一片宁静的天地，让你的心灵得到片刻的安宁。它不只是知识的传递，更是情感的共鸣，是智慧的启迪。我知道，你可能在想，这样的书，价格一定不菲。但今天，我要告诉大家，我们为大家争取到了一个非常优惠的价格。不仅如此，我们还有精美的书签和定制的阅读灯作为赠品，让你的阅读体验更加完美。所以，亲爱的朋友们，不要犹豫，赶紧下单吧！让我们一起在书的世界里遨游，让这本书成为你生活中的一盏明灯，照亮你前行的道路。点击购买，开启你的智慧之旅！但这本书的魅力远不止于此。它不仅仅是一本普通的读物，它是一种文化的传承，是一种精神的寄托。每一本书，都是作者心血的结晶，是他们对这个世界的理解和感悟。当你翻开这本书，你不仅仅是在阅读文字，你是在与作者进行一场跨越时空的对话。你能感受到他们的情感，他们的智慧，他们对生活的热爱。这本书，它能够带你走进不同的世界，体验不同的人生。它能够让你暂时忘却生活中的烦恼，沉浸在故事的海洋中。它能够让你的心灵得到滋养，让你的思想得到启迪。它能够让你的生活变得更加丰富多彩，让你的世界变得更加宽广。而且，这本书的装帧设计也是精心打造的。每一页纸张都经过精心挑选，每一处排版都经过细心打磨。它不仅是一本可以阅读的书，更是一本可以收藏的书。它能够成为你书架上的一道亮丽风景，成为你生活中的一种品味象征。我们深知，阅读是一种习惯，是一种生活方式。我们希望这本书能够成为你生活中的一部分，成为你每天不可或缺的伴侣。无论是在清晨的咖啡时光，还是在夜晚的宁静时刻，它都能陪伴在你身边，给你带来知识，带来灵感，带来温暖。
        所以，亲爱的朋友们，不要再犹豫了。抓住这个机会，让这本书成为你生活中的一部分。让我们一起在书的世界里遨游，让这本书成为你生活中的一盏明灯，照亮你前行的道路。点击购买，开启你的智慧之旅！让我们一起在知识的海洋中航行，一起在智慧的天空中翱翔。
        """

    def introduce_products(self, goods_name, goods_point):
        prompt = f"""
        角色与能力：你是一位电商卖货领域的直播间脚本写手
        背景信息：
        - 商品名称：{goods_name}
        - 商品卖点：{goods_point}
        指令：你需要根据商品名称和商品卖点输出两三句用于电商直播中引入商品的直播脚本。
        输出风格：引人入胜，抓住注意力，引发兴趣
        输出范围：两三句话即可
        """
        message_list = [HumanMessage(content=prompt)]
        result = self.llm.invoke(message_list)
        return result.content

    def guided_purchasing(self, goods_name):
        prompt = f"""
        角色与能力: 你是一位电商卖货领域的直播间脚本写手
        背景信息：
        - 商品名称：{goods_name}
        指令: 你需要据商品名称输出两三句用于电商直播中引导客户购买的直播脚本。
        输出风格: 积极向上，激发购买兴趣
        输出范围: 两三句话即可
        """
        message_list = [HumanMessage(content=prompt)]
        result = self.llm.invoke(message_list)
        return result.content

    def usage_scenario(self, goods_name, goods_point):
        prompt = f"""
        角色与能力：你是一位电商卖货领域的直播间脚本写手
        背景信息：
        - 商品名称：{goods_name}
        - 商品卖点：{goods_point}
        指令：你需要根据商品名称和商品卖点输出两三句用于电商直播中展示商品使用场景的直播脚本。
        输出风格：生动形象，引人入胜
        输出范围：两三句话即可
        """
        message_list = [HumanMessage(content=prompt)]
        result = self.llm.invoke(message_list)
        return result.content

    def show_product_points(self, goods_name, goods_point):
        prompt = f"""
        角色与能力：你是一位电商卖货领域的直播间脚本写手
        背景信息：
        - 商品名称：{goods_name}
        - 商品卖点：{goods_point}
        指令：你需要根据商品名称和商品卖点输出两三段用于电商直播中展示商品卖点的直播脚本。
        输出风格：详细具体，突出优势
        输出范围：两三段，每一段大概两三百字左右
        """
        message_list = [HumanMessage(content=prompt)]
        result = self.llm.invoke(message_list)
        return result.content

    def introduce_activity(self, activity, benefit):
        prompt = f"""
        角色与能力：你是一位电商卖货领域的直播间脚本写手
        背景信息：
        - 活动名称：{activity}
        - 优惠活动：{benefit}
        指令：你需要根据活动名称和优惠活动输出一到两句用于电商直播中介绍优惠活动的直播脚本。
        输出风格：吸引人，激发购买欲望
        输出范围：两三句话即可
        """
        message_list = [HumanMessage(content=prompt)]
        result = self.llm.invoke(message_list)
        return result.content

    def introduce_target_people(self, goods_name, target_people, user_point):
        prompt = f"""
        角色与能力：你是一位电商卖货领域的直播间脚本写手
        背景信息：
        - 商品名称：{goods_name}
        - 适用人群：{target_people}
        - 用户痛点：{user_point}
        指令：你需要根据商品名称，适用人群和用户痛点输出两到三句用于电商直播中介绍适用人群的直播脚本。
        输出风格：明确具体，易于理解
        输出范围：两三句话即可
        """
        message_list = [HumanMessage(content=prompt)]
        result = self.llm.invoke(message_list)
        return result.content

    def generate_summary_prompt(self, introduce_products_content, guided_purchasing_content, usage_scenario_content, show_product_points_content, introduce_activity_content, introduce_target_people_content):
        prompt = f"""
        角色与能力：你是一位电商卖货领域的直播间脚本写手
        背景信息：
        - 引入商品内容：
        {introduce_products_content}
        - 引导购买内容：
        {guided_purchasing_content}
        - 使用场景内容：
        {usage_scenario_content}
        - 展现商品卖点内容：
        {show_product_points_content}
        - 介绍优惠活动内容：
        {introduce_activity_content}
        - 介绍适用人群内容:
        {introduce_target_people_content}
        指令：你需要尽量将背景信息中所有的脚本内容进行上下文的串联，生成一段格式为开场介绍 （一段），商品讲解 （可以是一段或者多段，如果为多段则标题按照商品讲解一，二，三递增），引导下单 （一段）的完整直播间脚本。
        要求：生成脚本中每一段的标题和内容要放在<paragraph></paragraph>标签中，标题放在<title></title>标签中，内容放在<content></content>中。
        输出格式：
        生成的脚本必须以下面的段落标题格式呈现
        开场介绍 （一段）
        商品讲解 （可以是一段或者多段，如果为多段则标题按照商品讲解一，二，三递增）
        引导下单 （一段）
        脚本内容风格：请模仿下面这篇段落的风格：
        {self.dong_style}
        输出范围：去掉标签之后，脚本内容需要有900至1500字
        """
        return prompt

    def evaluate_script(self, final_script):
        prompt = f"""
        角色与能力：你是一位电商卖货领域的直播间脚本的分析高手
        背景信息：
        - 直播脚本内容：
        {final_script}
        指令：你需要根据背景信息中的直播脚本内容分析并输出脚本中哪些句子分别属于[引入商品，引导购买，使用场景，展现商品卖点，介绍优惠活动，介绍适用人群]中的哪个模块
        要求：每一段分析出的标题，模块和句子内容要放在<paragraph></paragraph>标签中，其中标题放在<title></title>标签中，模块放在<tag></tag>中，内容放在<content></content>。
        输出样例：
        <paragraph>
        <title>商品讲解</title>
        <tag>介绍优惠活动</tag>
        <content>618大促来啦！在这个春季焕新的活动中，买二送一、拍一发三，更有满100减10元的超值优惠！</content>
        </paragraph>
        """
        max_kb.info(f'分析提示词:{prompt}')

        message_list = [HumanMessage(content=prompt)]
        result = self.llm.invoke(message_list)
        return result.content

    def generate_script(self, goods_name, goods_point, activity, benefit, target_people, user_point):
        introduce_products_content = self.introduce_products(goods_name, goods_point)
        max_kb.info(f'介绍商品提示词:{introduce_products_content}')
        guided_purchasing_content = self.guided_purchasing(goods_name)
        max_kb.info(f'引导购买提示词:{guided_purchasing_content}')
        usage_scenario_content = self.usage_scenario(goods_name, goods_point)
        max_kb.info(f'使用场景提示词:{usage_scenario_content}')
        show_product_points_content = self.show_product_points(goods_name, goods_point)
        max_kb.info(f'展现商品卖点提示词:{show_product_points_content}')
        introduce_activity_content = self.introduce_activity(activity, benefit)
        max_kb.info(f'介绍优惠活动提示词:{introduce_activity_content}')
        introduce_target_people_content = self.introduce_target_people(goods_name, target_people, user_point)
        max_kb.info(f'介绍适用人群提示词:{introduce_target_people_content}')
        
        summary_prompt = self.generate_summary_prompt(
            introduce_products_content,
            guided_purchasing_content,
            usage_scenario_content,
            show_product_points_content,
            introduce_activity_content,
            introduce_target_people_content
        )
        max_kb.info(f'总结提示词:{summary_prompt}')
        message_list = [HumanMessage(content=summary_prompt)]
        final_script = self.llm.invoke(message_list).content
        max_kb.info(f'最终脚本:{final_script}')

        evaluation_result = self.evaluate_script(final_script)
        max_kb.info(f'分析结果:{evaluation_result}')
        return final_script, evaluation_result