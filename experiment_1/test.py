import openai
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=0.5)
    parser.add_argument('--vote_times', type=int, default=5)
    parser.add_argument('--max_tokens', type=int, default=10)
    parser.add_argument('--api_key', type=str, required=True)

    args = parser.parse_args()

    openai.api_key = args.api_key

    model = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Hello, world!"}
        ]
    )

    print(model)



if __name__ == "__main__":
    main()
