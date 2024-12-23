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
        self.jiaqi_style = """
        各位亲爱的观众朋友们，大家好！非常感谢大家在百忙之中抽出时间来到我的直播间。今天，我要向大家介绍的这款产品，是我亲自试用过，并且效果让我惊艳的一款护肤品。它不仅能够深层滋养你的肌肤，还能够让你的肌肤焕发出自然的光泽。

        首先，我们来谈谈这款产品的外观设计。它的包装非常精致，采用了高级的材质，拿在手里你就能感觉到它的与众不同。这种设计不仅仅是为了美观，更是为了保护产品的有效成分，确保你每次使用时都能享受到最佳的效果。

        当我们打开包装，你会看到这款护肤品的质地非常细腻，它采用了最新的乳化技术，使得产品能够迅速被肌肤吸收，而不会给肌肤带来任何负担。你可以看到，我手上试用后，肌肤立刻变得水润光滑，而且没有任何油腻感。

        这款产品的成分表也非常让人放心。它富含多种天然植物精华，这些成分都是经过精心挑选的，旨在为你的肌肤提供最温和的护理。无论你是敏感肌肤，还是混合性肌肤，这款产品都能给你带来舒适的使用体验，绝不会引起任何不适。

        在价格方面，我为大家争取到了一个非常优惠的价格。相比市面上的其他同类产品，我们的性价比绝对是最高的。今天购买，你不仅能享受到优惠的价格，还能获得额外的赠品，这样的好事可不是天天都有的。

        所以，亲爱的朋友们，如果你还在为选择哪款护肤品而犹豫不决，那么今天就是你做出决定的最佳时机。我们的库存有限，机会稍纵即逝，所以一定要抓紧时间下单哦。

        好了，不多说了，让我们立刻开启抢购模式。准备好了吗？请大家密切关注屏幕，我将马上为大家上链接。记得，手快有，手慢无，赶紧行动起来，让我们一起享受美丽肌肤带来的自信和快乐吧！
        """

        self.normal_style = """
        大家好，欢迎来到我们的直播间，我是你们的好朋友！今天给大家带来的不仅仅是一双鞋，而是一种全新的跑步体验——我们的超级跑鞋！这可不是普通的跑鞋，它结合了最新的科技和设计理念，让你的每一次跑步都变得轻松愉快。

        首先，让我们来聊聊这双鞋的外观设计。它采用了流线型的设计，不仅时尚，还能减少空气阻力，让你在跑步时更加轻盈。而且，颜色搭配也非常抢眼，无论是在跑道上还是街头，都能成为众人瞩目的焦点。

        接下来，说说它的材质。我们选用的是顶级的透气材料，即使在长时间的运动后，你的双脚也能保持干爽舒适。而且，鞋底采用了特殊的防滑橡胶，无论是湿滑的雨天还是复杂的地形，都能给你提供稳固的抓地力。

        当然，最重要的还是舒适度。我们的跑鞋内部空间宽敞，适合各种脚型，而且鞋垫采用了记忆棉材质，能够根据你的脚型进行自适应，提供最佳的支撑和缓冲。

        现在，我知道你们已经迫不及待想要拥有这样一双完美的跑鞋了。好消息是，今天我们直播间有特别优惠，前100名下单的朋友将获得额外的折扣，而且还有限量版赠品等你来抢！

        别犹豫了，这双跑鞋不仅能提升你的跑步体验，更是你时尚生活的点睛之笔。点击屏幕下方的购买链接，立刻将它带回家。数量有限，先到先得哦！赶紧行动起来，让我们一起跑出健康，跑出风采！

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
        输出范围：两三段，根据背景信息中的商品卖点的内容长度控制字数为200-1000字上下
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

    def generate_summary_prompt(
        self,
        introduce_products_content,
        guided_purchasing_content,
        usage_scenario_content,
        show_product_points_content,
        introduce_activity_content,
        introduce_target_people_content,
        script_style,
    ):
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
        {script_style}
        输出范围：长度需在900-1500字之间
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
        message_list = [HumanMessage(content=prompt)]
        result = self.llm.invoke(message_list)
        return result.content

    def continuity_phrases(self, final_script):
        prompt = f"""
        角色与能力：你是一位电商卖货领域的直播间脚本的连贯高手
        背景信息：
        - 直播脚本内容：
        {final_script}
        指令：你需要以背景信息中的直播脚本内容为基础，假如播放完每一段内容后需要回答用户问题，问答完之后需要有一到两句连贯的话用于引出下面的段落，请你给每两段之间都生成一至两句连贯的话
        输出风格：跟直播脚本内容风格保持一致
        要求：只输出两段之间连贯的话，不需要带上原先的脚本内容，每输出一组连贯的话，请把这段话上文的标题和连贯的句子要放在<paragraph></paragraph>标签中，其中标题放在<title></title>标签中，连贯句子放在<content></content>。
        输出样例：
        <paragraph>
        <title>商品讲解</title>
        <content>现在，让我们回到刚才的话题，继续为您详细介绍。</content>
        </paragraph>
        """

        message_list = [HumanMessage(content=prompt)]
        result = self.llm.invoke(message_list)
        return result.content

    def generate_ssml(self, text):
        prompt = f"""
        角色与能力：你是一位电商卖货领域的直播间脚本写手  
        背景信息：
        - 口播文案：{text}
        指令：你需要给背景信息中的口播文案中<content></content>标签里的文案内容适当的加上一些ssml标签，使其更符合语音播报，直播带货的要求，直播效果更好。
        输出格式 ：口播文案输入的内容和框架格式不变，仅在<content></content>标签里做ssml标签的修改，其他内容不用动。
        """

        message_list = [HumanMessage(content=prompt)]
        result = self.llm.invoke(message_list)
        return result.content

    def generate_comments(self, comments,choose_nums,goods_info):
        prompt = f"""
        角色与能力：你是一位电商卖货领域的直播间弹幕主播。
        背景信息：
        - 弹幕：{comments}
        - 商品信息：{goods_info}
        指令：你需要根据背景信息中的弹幕，挑选出至多{choose_nums}条与直播的商品信息的弹幕，如果背景信息弹幕为空则输出空。
        输出格式 ：只输出挑选出来的弹幕，将每条弹幕都分别放在<content></content>中。
        输出样例：
        <content>这款鞋卖多少钱</content>
        <content>这款鞋耐穿吗</content>
        """

        message_list = [HumanMessage(content=prompt)]
        result = self.llm.invoke(message_list)
        return result.content
    
    def beauty_comments(self, script_style, question, answer, paragraph_now):
        prompt = f"""
        角色与能力：你是一位电商卖货领域的直播主播。
        背景信息：
        - 弹幕问题：{question}
        - 答案：{answer}
        - 当前段落：{paragraph_now}
        直播风格：请模仿下面这篇段落的风格：
        {script_style}
        指令：假设你现在在直播，当介绍完当前商品的段落之后你需要对弹幕进行回答，现在请你结合背景信息中的观众问题对答案进行润色，并使其背景信息中的当前段落语义连贯。
        输出格式 ：将润色后的回答内容放在<data></data>中。
        """

        message_list = [HumanMessage(content=prompt)]
        max_kb.info(f"润色回答提示词{message_list}")
        result = self.llm.invoke(message_list)
        max_kb.info(f"润色回答结果{result.content}")
        return result.content
    def generate_script(
        self,
        goods_name,
        goods_point,
        activity,
        benefit,
        target_people,
        user_point,
        style,
        is_ssml_open,
    ):
        introduce_products_content = self.introduce_products(goods_name, goods_point)
        max_kb.info(f"介绍商品提示词:{introduce_products_content}")
        guided_purchasing_content = self.guided_purchasing(goods_name)
        max_kb.info(f"引导购买提示词:{guided_purchasing_content}")
        usage_scenario_content = self.usage_scenario(goods_name, goods_point)
        max_kb.info(f"使用场景提示词:{usage_scenario_content}")
        show_product_points_content = self.show_product_points(goods_name, goods_point)
        max_kb.info(f"展现商品卖点提示词:{show_product_points_content}")
        introduce_activity_content = self.introduce_activity(activity, benefit)
        max_kb.info(f"介绍优惠活动提示词:{introduce_activity_content}")
        introduce_target_people_content = self.introduce_target_people(
            goods_name, target_people, user_point
        )
        max_kb.info(f"介绍适用人群提示词:{introduce_target_people_content}")
        script_style = self.normal_style
        if style == 2:
            script_style = self.dong_style
        if style == 3:
            script_style = self.jiaqi_style
        summary_prompt = self.generate_summary_prompt(
            introduce_products_content,
            guided_purchasing_content,
            usage_scenario_content,
            show_product_points_content,
            introduce_activity_content,
            introduce_target_people_content,
            script_style,
        )
        max_kb.info(f"总结提示词:{summary_prompt}")
        message_list = [HumanMessage(content=summary_prompt)]
        final_script = self.llm.invoke(message_list).content
        max_kb.info(f"最终脚本:{final_script}")

        continuity_phrases = self.continuity_phrases(final_script)
        max_kb.info(f"连贯句子:{continuity_phrases}")

        evaluation_result = self.evaluate_script(final_script)
        max_kb.info(f"分析结果:{evaluation_result}")

        if is_ssml_open:
            # 生成最终脚本的 SSML
            final_script_ssml = self.generate_ssml(final_script)
            max_kb.info(f"最终脚本 SSML:{final_script_ssml}")

            # 生成连贯句子的 SSML
            continuity_phrases_ssml = self.generate_ssml(continuity_phrases)
            max_kb.info(f"连贯句子 SSML:{continuity_phrases_ssml}")

            return final_script_ssml, evaluation_result, continuity_phrases_ssml
        return final_script, evaluation_result, continuity_phrases

    def generate_comment(self,comments,choose_num,goods_info):
        comments_content = self.generate_comments(comments,choose_num,goods_info)
        return comments_content

    def beauty_comment(self,style, question, answer, paragraph_now):
        script_style = self.normal_style
        if style == 2:
            script_style = self.dong_style
        if style == 3:
            script_style = self.jiaqi_style
        comments_content = self.beauty_comments(script_style, question, answer, paragraph_now)
        return comments_content
