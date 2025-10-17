from util.construct_email import send_email
from arxiv_daily import ArxivDaily
import argparse
import os
import sys

if __name__ == "__main__":
    '''
    <Examples>

    (1) When sending via Slack
    python main.py \
    --categories cs.CL cs.LG \
    --provider OpenAI --model gpt-4o \
    --base_url https://api.openai.com/v1 \
    --api_key sk-xxxxx \
    --filter_method splade \
    --language korean \
    --notify_method slack \
    --slack_webhook_url https://hooks.slack.com/services/AAA/BBB/CCC


    (2) When sending via email
    python main.py \
    --categories cs.AI cs.SD \
    --provider OpenAI --model gpt-4o \
    --base_url https://api.openai.com/v1 \
    --api_key sk-xxxxx \
    --filter_method bm25 \
    --language english \
    --notify_method email \
    --smtp_server smtp.gmail.com \
    --smtp_port 587 \
    --sender your_email@gmail.com \
    --receiver target_email@gmail.com \
    --sender_password your_password

    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arxiv Daily")

    # Basic settings
    parser.add_argument("--categories", nargs="+", required=True, help="arXiv categories")
    parser.add_argument("--max_paper_num", type=int, default=60)
    parser.add_argument("--max_entries", type=int, default=100)
    parser.add_argument("--provider", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--save", action="store_true", help="Save email/slack content to file")
    parser.add_argument("--save_dir", type=str, default="./arxiv_history")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--description", type=str, default="description.txt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--title", type=str, default="Daily arXiv")

    # Option 1: Select filtering method
    parser.add_argument(
        "--filter_method",
        type=str,
        choices=["bm25", "dpr", "splade", "none"],
        default="none",
        help="Filtering method to use (bm25, dpr, splade, none)"
    )

    # Option 2: Choose final output language
    parser.add_argument(
        "--language",
        type=str,
        choices=["korean", "english", "chinese"],
        default="korean",
        help="Language for final summaries (korean, english, chinese)"
    )

    # Option 3: Notification method (Slack / Email)
    parser.add_argument(
        "--notify_method",
        type=str,
        choices=["slack", "email"],
        required=True,
        help="Notification method: slack or email"
    )

    # Option 4: Arguments for Slack
    parser.add_argument(
        "--slack_webhook_url",
        type=str,
        default=None,
        help="Slack webhook URL (required if notify_method=slack)"
    )

    # Option 5: Arguments for Email
    parser.add_argument("--smtp_server", type=str, help="SMTP server address")
    parser.add_argument("--smtp_port", type=int, help="SMTP port number")
    parser.add_argument("--sender", type=str, help="Sender email address")
    parser.add_argument("--receiver", type=str, help="Receiver email address")
    parser.add_argument("--sender_password", type=str, help="Sender email password or app password")




    args = parser.parse_args()

    ###--------------------------------------------------------------------------------###

    # Validate provider type
    if args.provider.lower() not in ["ollama", "openai", "siliconflow"]:
        raise ValueError("Unsupported provider. Choose from Ollama, OpenAI, or SiliconFlow.")
    
    # OpenAI and SiliconFlow require base_url and api_key
    if args.provider.lower() != "ollama":
        assert args.base_url is not None, "base_url is required for OpenAI and SiliconFlow"
        assert args.api_key is not None, "api_key is required for OpenAI and SiliconFlow"


    # Load research description text
    with open(args.description, "r", encoding="utf-8") as f:
        args.description = f.read()

    # Test LLM availability
    if args.provider.lower() == "ollama":
        from llm.Ollama import Ollama

        try:
            model = Ollama(args.model)
            model.inference("Hello, who are you?")
        except Exception as e:
            print(e)
            assert False, "Model not initialized successfully."
    elif (
        args.provider == "OpenAI"
        or args.provider == "openai"
        or args.provider == "SiliconFlow"
    ):
        from llm.GPT import GPT

        try:
            model = GPT(args.model, args.base_url, args.api_key)
            model.inference("Hello, who are you?")
        except Exception as e:
            print(e)
            assert False, "Model not initialized successfully."
    else:
        assert False, "Model not supported."

    ###--------------------------------------------------------------------------------###

    # Create directory to save results (if enabled)
    if args.save:
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        args.save_dir = None

    # Initialize ArxivDaily object
    arxiv_daily = ArxivDaily(
        categories=args.categories,
        max_entries=args.max_entries,
        max_paper_num=args.max_paper_num,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        description=args.description,
        num_workers=args.num_workers,
        temperature=args.temperature,
        save_dir=args.save_dir,
        filter_method=args.filter_method,
        language=args.language,
    )

    # Send notifications based on selected method
    if args.notify_method == "slack":
        # Slack mode requires the webhook URL
        if not args.slack_webhook_url:
            print("Error: --slack_webhook_url is required when using Slack.")
            sys.exit(1)
        arxiv_daily.send_slack(args.slack_webhook_url, title=f"Arxiv Daily Summary ({args.language})")

    elif args.notify_method == "email":
        # Email mode requires all SMTP parameters
        required_fields = [
            args.smtp_server,
            args.smtp_port,
            args.sender,
            args.receiver,
            args.sender_password,
        ]
        if not all(required_fields):
            print("Error: Missing SMTP or email parameters for email sending.")
            sys.exit(1)

        arxiv_daily.send_email(
            args.sender,
            args.receiver,
            args.sender_password,
            args.smtp_server,
            args.smtp_port,
            args.title,
        )


    
