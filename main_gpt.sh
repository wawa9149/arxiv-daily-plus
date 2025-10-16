cd /path/to/customized-arxiv-daily
python main.py --categories cs.CV cs.AI cs.CL cs.CR cs.LG \
    --provider OpenAI --model gpt-4o \
    --base_url https://api.openai.com/v1 --api_key * \
    --smtp_server smtp.qq.com --smtp_port 465 \
    --sender * --receiver * \
    --sender_password * \
    --num_workers 16 \
    --temperature 0.7 \
    --title "Daily arXiv" \
    --description "description.txt" \
    --save