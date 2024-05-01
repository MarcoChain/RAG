import gradio as gr

from text_utils import (
    prepare_text,
    open_and_read_pdfs,
    
)

from rag_components_utils import(
    get_device,
    load_llm,
    generate_reply,
    create_and_store_embeddings,
    load_re_ranking_model,
    load_embedding_model,
    retrieve_topk_results,
)

device, attn_implementation, quantization_config = get_device()

pages_and_texts = open_and_read_pdfs(
        data_path = "./data",
    )
pages_and_chunks_over_min_token_len = prepare_text(
        pages_and_texts = pages_and_texts,
        num_sentence_chunk_size = 10,
        min_token_length = 30
    )


embedding_model = load_embedding_model(
    embedding_model_name = "all-mpnet-base-v2",
    device = device
)
faiss_dataset = create_and_store_embeddings(
        pages_and_chunks_over_min_token_len = pages_and_chunks_over_min_token_len,
        embedding_model = embedding_model,
        batch_size = 512,
        csv_path = "./csv",
        top_k_context = 5,
        device = device
    )

re_ranking_model = load_re_ranking_model(
    re_ranking_model_name = "mixedbread-ai/mxbai-rerank-large-v1",
    device = device
)

tokenizer, llm_model = load_llm(
    llm_model_name = "google/gemma-1.1-2b-it",
    attn_implementation = attn_implementation,
    low_cpu_mem_usage = False,
    quantization_config = quantization_config,
    token = None,#"<YOUR TOKEN HERE>",
    device = device
)

with gr.Blocks() as demo:
    gr.Markdown("## Chat with ZAG")
    with gr.Column():
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column():
                message = gr.Textbox(label="Chat Message Box", placeholder="Message ZAG", show_label=False)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    clear = gr.Button("Clear")

    def respond(query, chat_history):
        # convert chat history to prompt
        converted_chat_history = ""
        if len(chat_history) > 0:
          for c in chat_history:
            converted_chat_history += f"<|prompter|>{c[0]}<|endoftext|><|assistant|>{c[1]}<|endoftext|>"

        # send request to endpoint
        selected_text, _ = retrieve_topk_results(
            query = query,
            faiss_dataset = faiss_dataset,
            embedding_model = embedding_model,
            re_ranking_model = re_ranking_model,
            top_k_context = 5
        )

        outputs_decoded = generate_reply(
            llm_model = llm_model,
            tokenizer = tokenizer,
            query = query,
            selected_text = selected_text,
            device = device
        )
        chat_history.append((query, "ZAG:\n" + outputs_decoded))
        return "", chat_history

    submit.click(respond, [message, chatbot], [message, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
