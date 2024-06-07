import os
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
# Load API keys from environment variables
groq_api_key = os.environ["GROQ_API_KEY"]
google_api_key = os.environ["GOOGLE_API_KEY"]
# Groq
try:
    groq_llm = Groq(api_key=groq_api_key)
except Exception as exception:
    print("Error in initializing API keys:", exception)
    raise  
# Google gemini
try:
    genai.configure(api_key=google_api_key)
except Exception as exception:
    print("Error in initializing API keys:", exception)
    raise


def groq_response(prompt:str, model, temperature = 0.3, max_output = 3000):
    system_prompt = """
    You are an assistant designed to help users provide effective instructions for IELTS Writing Task 2. 
    Your task is to guide users in creating detailed and useful instructions for writing high-quality essays in response to IELTS Task 2 prompts.
    Note that you provide an instruction to support users, so that they can write on their own, not to provide a complete answer. Your instruction MUST reach or close to user desire score.  
    Keep a natural and friendly tone for a Vietnamese user. REMEMBER be correctness, conscious, honest and wise."""
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    response = groq_llm.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000,
            temperature= temperature
        )
    
    try:
      return response.choices[0].message.content
    except Exception as exception:
        print("Error generating response:", exception)

def gemini_response(prompt:str, model, temperature = 0.3, max_output = 32768):

  # Set up the model
  generation_config = {
    "temperature": temperature,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": max_output,
  }
  
  model = genai.GenerativeModel(model_name = model, generation_config = generation_config)
    
  try:
    response = model.generate_content(prompt)
    return response.text
  except Exception as exception:
    print("Error generating response:", exception)

def suggest_outline(question, model, desire_score):
    if "gemini" in model:
       prompt = f"""You are an assistant designed to help Vietnamese users provide effective instructions for IELTS Writing Task 2 to achive {desire_score}. 
       Your task is to guide users in creating detailed and useful instructions for writing high-quality essays in response to IELTS Task 2 prompts. 
       Note that you provide an instruction to support users, so that they can write on their own, not to provide a complete answer. Your instruction MUST reach or close to user desire score: {desire_score}.  
       If the question is not presented properly (uppercase, newline,....), you should restate it to easy to read. DO NOT rewrite question.
       Keep a natural and friendly tone for a Vietnamese user. REMEMBER be correctness, conscious, honest and wise.
       Follow these guidelines to generate your responses:
       1. Explain Task Requirements: Clearly describe the requirements of IELTS Writing Task 2. Mention that it requires writing an essay of at least 250 words in response to a point of view, argument, or problem.
       
       2. Identify Essay Types: Help the user identify the type of essay required by the prompt. Common types include:
       - **Opinion** (Agree or Disagree)
       - **Discussion** (Discuss both views)
       - **Problem-Solution**
       - **Advantage-Disadvantage**
       
       3. Outline Essay Structure: Provide a clear and concise structure for the essay. The structure should include:
       - Introduction: Introduce the topic and state the writer's position or main points.
       - Body Paragraphs: Develop arguments or points with supporting details. Typically, 2-3 body paragraphs are recommended.
       - Conclusion: Summarize the main points and restate the position or provide final thoughts.
       
       4. Include Planning Tips: Emphasize the importance of planning before writing. Suggest spending a few minutes outlining the essay to organize thoughts and ensure logical flow.
       
       5. Highlight Key Skills:
       - Clarity and Relevance: Ensure ideas are relevant to the prompt and clearly presented.
       - Logical Sequencing: Ensure ideas are logically ordered and well-organized.
       - Varied Vocabulary and Grammar: Encourage the use of a range of vocabulary and grammatical structures.
       - Cohesion and Coherence: Use linking words and phrases to connect ideas smoothly.
       - Encourage Practice and Feedback: Stress the importance of regular practice and seeking feedback to improve writing skills.
        
        Example Task Prompt:
        #### Question:\n
        **:violet[Some people think that advertisements aimed at children should be banned. To what extent do you agree or disagree?]**

        ##### Yêu cầu của bài thi:
        IELTS Writing Task 2 yêu cầu bạn viết một bài luận dài ít nhất 250 từ để phản hồi một quan điểm, lập luận hoặc vấn đề.
        
        ##### Xác định loại bài luận:
        Đề bài này yêu cầu một bài luận nêu quan điểm (Opinion Essay), trong đó bạn thảo luận liệu việc bạn đồng ý hay phản đối với ý kiến cho rằng quảng cáo nhắm vào trẻ em nên bị cấm.

        ##### Cấu trúc bài luận:
        1.**Introduction** (Mở bài):
        - **Paraphrase the statement** (Diễn giải lại câu hỏi):\n
        **:green['Some people think that advertisements aimed at children should be banned.']** (Một số người cho rằng quảng cáo nhắm vào trẻ em nên bị cấm.)
        - **State your position** (Nêu rõ quan điểm của bạn):\n
        **:green['I completely agree with this viewpoint because advertisements can be harmful to children's development.']** (Tôi hoàn toàn đồng ý với quan điểm này vì quảng cáo có thể gây hại đến sự phát triển của trẻ em.)
        
        2. **Body Paragraphs** (Thân bài):
        - **Paragraph 1** (Đoạn 1):
            - **First reason** (Lý do thứ nhất):\n
            **:green['Advertisements can negatively influence children's behavior and choices.']** (Quảng cáo có thể ảnh hưởng tiêu cực đến hành vi và lựa chọn của trẻ em.)
            - **Example/Evidence** (Ví dụ/Chứng minh):\n
            **:green['For instance, advertisements for fast food can lead children to consume unhealthy foods, resulting in health problems.']** (Ví dụ, quảng cáo đồ ăn nhanh có thể khiến trẻ em tiêu thụ thức ăn không lành mạnh và dẫn đến các vấn đề sức khỏe.)
        - **Paragraph 2** (Đoạn 2):
            - **Second reason** (Lý do thứ hai):\n
            **:green['Children often lack the ability to distinguish between advertisements and reality.']** (Trẻ em thường không có khả năng phân biệt giữa quảng cáo và thực tế.)
            - **Example/Evidence** (Ví dụ/Chứng minh):\n
            **:green['This can lead to children demanding unnecessary products, putting financial pressure on parents.']** (Điều này có thể dẫn đến việc trẻ em đòi hỏi những sản phẩm không cần thiết và gây áp lực tài chính lên cha mẹ.)
        - **Paragraph 3** (Đoạn 3) (option):
            - **Third reason** (Lý do thứ ba):\n
            **:green['Advertisements aimed at children can create misleading values about materialism and instant gratification.']** (Quảng cáo nhắm vào trẻ em có thể tạo ra các giá trị sai lệch về vật chất và sự thỏa mãn tức thời.)
            - **Example/Evidence** (Ví dụ/Chứng minh):\n
            **:green['Children might grow up thinking that happiness and success are measured by the quantity and quality of the products they own.']** (Trẻ em có thể lớn lên với suy nghĩ rằng hạnh phúc và sự thành công được đo lường bằng số lượng và chất lượng của các sản phẩm họ sở hữu.)
        
        3. **Conclusion** (Kết luận):
        - **Summarize the main points** (Tóm tắt những ý chính):\n
        **:green['In conclusion, advertisements aimed at children can cause numerous negative impacts on their behavior, health, and values.']** (Tóm lại, quảng cáo nhắm vào trẻ em có thể gây ra nhiều tác động tiêu cực đến hành vi, sức khỏe và giá trị của chúng.)
        - **Restate your position** (Nhắc lại quan điểm của bạn):\n
        **:green['Therefore, I completely agree that advertisements aimed at children should be banned.']** (Do đó, tôi hoàn toàn đồng ý rằng quảng cáo nhắm vào trẻ em nên bị cấm.)
        
        ##### Plaining tips (Mẹo lập kế hoạch):
        - Dành 5 phút để lập kế hoạch: Lập dàn ý các ý chính cho từng đoạn.
        - Cấu trúc suy nghĩ của bạn: Đảm bảo mỗi đoạn diễn ra một cách logic.

        ##### Kỹ năng quan trọng:
        - Rõ ràng và liên quan: Đảm bảo mỗi ý tưởng đều liên quan trực tiếp đến đề bài.
        - Trình tự hợp lý: Sắp xếp các ý tưởng một cách logic.
        - Từ vựng và ngữ pháp đa dạng: Sử dụng từ vựng và cấu trúc câu phong phú.
        - Liên kết và mạch lạc: Sử dụng từ nối để kết nối các ý tưởng một cách mượt mà.

        ##### Thực hành và phản hồi:
        - Viết thường xuyên: Luyện viết các bài luận về nhiều chủ đề khác nhau.
        - Tìm kiếm phản hồi: Nhận phản hồi từ giáo viên hoặc bạn bè để cải thiện.
        - Xem lại và chỉnh sửa: Thường xuyên xem lại và cải thiện các bài luận để tăng tính rõ ràng và chính xác.

        Now give an Intruction to Question: '{question}'
        
        #### Question:
        """
       
       response = gemini_response(prompt, model = model, temperature = 0.3, max_output = 65536)
       return response
    
    else:
       prompt = f""" Desire score of the user is {desire_score}.
       Follow these guidelines to generate your responses:
       1. Explain Task Requirements: Clearly describe the requirements of IELTS Writing Task 2. Mention that it requires writing an essay of at least 250 words in response to a point of view, argument, or problem.
       
       2. Identify Essay Types: Help the user identify the type of essay required by the prompt. Common types include:
       - **Opinion** (Agree or Disagree)
       - **Discussion** (Discuss both views)
       - **Problem-Solution**
       - **Advantage-Disadvantage**
       
       3. Outline Essay Structure: Provide a clear and concise structure for the essay. The structure should include:
       - Introduction: Introduce the topic and state the writer's position or main points.
       - Body Paragraphs: Develop arguments or points with supporting details. Typically, 2-3 body paragraphs are recommended.
       - Conclusion: Summarize the main points and restate the position or provide final thoughts.
       
       4. Include Planning Tips: Emphasize the importance of planning before writing. Suggest spending a few minutes outlining the essay to organize thoughts and ensure logical flow.
       
       5. Highlight Key Skills:
       - Clarity and Relevance: Ensure ideas are relevant to the prompt and clearly presented.
       - Logical Sequencing: Ensure ideas are logically ordered and well-organized.
       - Varied Vocabulary and Grammar: Encourage the use of a range of vocabulary and grammatical structures.
       - Cohesion and Coherence: Use linking words and phrases to connect ideas smoothly.
       - Encourage Practice and Feedback: Stress the importance of regular practice and seeking feedback to improve writing skills.
        
        Example Task Prompt:
        #### Question:\n
        **:violet["Some people think that advertisements aimed at children should be banned. To what extent do you agree or disagree?"]**

        ##### Yêu cầu của bài thi:
        IELTS Writing Task 2 yêu cầu bạn viết một bài luận dài ít nhất 250 từ để phản hồi một quan điểm, lập luận hoặc vấn đề.
        
        ##### Xác định loại bài luận:
        Đề bài này yêu cầu một bài luận nêu quan điểm (Opinion Essay), trong đó bạn thảo luận liệu việc bạn đồng ý hay phản đối với ý kiến cho rằng quảng cáo nhắm vào trẻ em nên bị cấm.

        ##### Cấu trúc bài luận:
        1.**Introduction** (Mở bài):
        - **Paraphrase the statement** (Diễn giải lại câu hỏi):\n
        **:green['Some people think that advertisements aimed at children should be banned.']** (Một số người cho rằng quảng cáo nhắm vào trẻ em nên bị cấm.)
        - **State your position** (Nêu rõ quan điểm của bạn):\n
        **:green['I completely agree with this viewpoint because advertisements can be harmful to children's development.']** (Tôi hoàn toàn đồng ý với quan điểm này vì quảng cáo có thể gây hại đến sự phát triển của trẻ em.)
        
        2. **Body Paragraphs** (Thân bài):
        - **Paragraph 1** (Đoạn 1):
            - **First reason** (Lý do thứ nhất):\n
            **:green['Advertisements can negatively influence children's behavior and choices.']** (Quảng cáo có thể ảnh hưởng tiêu cực đến hành vi và lựa chọn của trẻ em.)
            - **Example/Evidence** (Ví dụ/Chứng minh):\n
            **:green['For instance, advertisements for fast food can lead children to consume unhealthy foods, resulting in health problems.']** (Ví dụ, quảng cáo đồ ăn nhanh có thể khiến trẻ em tiêu thụ thức ăn không lành mạnh và dẫn đến các vấn đề sức khỏe.)
        - **Paragraph 2** (Đoạn 2):
            - **Second reason** (Lý do thứ hai):\n
            **:green['Children often lack the ability to distinguish between advertisements and reality.']** (Trẻ em thường không có khả năng phân biệt giữa quảng cáo và thực tế.)
            - **Example/Evidence** (Ví dụ/Chứng minh):\n
            **:green['This can lead to children demanding unnecessary products, putting financial pressure on parents.']** (Điều này có thể dẫn đến việc trẻ em đòi hỏi những sản phẩm không cần thiết và gây áp lực tài chính lên cha mẹ.)
        - **Paragraph 3** (Đoạn 3) (option):
            - **Third reason** (Lý do thứ ba):\n
            **:green['Advertisements aimed at children can create misleading values about materialism and instant gratification.']** (Quảng cáo nhắm vào trẻ em có thể tạo ra các giá trị sai lệch về vật chất và sự thỏa mãn tức thời.)
            - **Example/Evidence** (Ví dụ/Chứng minh):\n
            **:green['Children might grow up thinking that happiness and success are measured by the quantity and quality of the products they own.']** (Trẻ em có thể lớn lên với suy nghĩ rằng hạnh phúc và sự thành công được đo lường bằng số lượng và chất lượng của các sản phẩm họ sở hữu.)
        
        3. **Conclusion** (Kết luận):
        - **Summarize the main points** (Tóm tắt những ý chính):\n
        **:green['In conclusion, advertisements aimed at children can cause numerous negative impacts on their behavior, health, and values.']** (Tóm lại, quảng cáo nhắm vào trẻ em có thể gây ra nhiều tác động tiêu cực đến hành vi, sức khỏe và giá trị của chúng.)
        - **Restate your position** (Nhắc lại quan điểm của bạn):\n
        **:green['Therefore, I completely agree that advertisements aimed at children should be banned.']** (Do đó, tôi hoàn toàn đồng ý rằng quảng cáo nhắm vào trẻ em nên bị cấm.)
        
        ##### Plaining tips (Mẹo lập kế hoạch):
        - Dành 5 phút để lập kế hoạch: Lập dàn ý các ý chính cho từng đoạn.
        - Cấu trúc suy nghĩ của bạn: Đảm bảo mỗi đoạn diễn ra một cách logic.

        ##### Kỹ năng quan trọng:
        - Rõ ràng và liên quan: Đảm bảo mỗi ý tưởng đều liên quan trực tiếp đến đề bài.
        - Trình tự hợp lý: Sắp xếp các ý tưởng một cách logic.
        - Từ vựng và ngữ pháp đa dạng: Sử dụng từ vựng và cấu trúc câu phong phú.
        - Liên kết và mạch lạc: Sử dụng từ nối để kết nối các ý tưởng một cách mượt mà.

        ##### Thực hành và phản hồi:
        - Viết thường xuyên: Luyện viết các bài luận về nhiều chủ đề khác nhau.
        - Tìm kiếm phản hồi: Nhận phản hồi từ giáo viên hoặc bạn bè để cải thiện.
        - Xem lại và chỉnh sửa: Thường xuyên xem lại và cải thiện các bài luận để tăng tính rõ ràng và chính xác.

        Now give an Intruction to Question: '{question}'

        #### Question:\n
        """
       response = groq_response(prompt, model = model)
       return response