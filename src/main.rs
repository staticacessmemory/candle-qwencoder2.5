use std::{fs::File, io::{Write, stdout}, time::Instant};

use candle_core::{Device, Tensor, quantized::gguf_file};
use candle_transformers::{generation::{LogitsProcessor, Sampling}, utils::apply_repeat_penalty};
use clap::Parser;
use hf_hub::{Repo, api::sync::Api};

use crate::tokenoutputstream::TokenOutputStream;

mod tokenoutputstream;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    prompt: Option<String>,

    #[arg(short = 'n', long, default_value_t = 1000)]
    sample_len: usize,

    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long)]
    top_k: Option<usize>,

    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    #[arg(long, default_value = "false")]
    split_prompt: bool,

    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    model: Option<String>,
    
    tokenizer: Option<String>,
}

impl Args {
    fn tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer>{
        let repo = String::from("Qwen/Qwen2.5-Coder-1.5B-Instruct");
        
        let api = Api::new()?.model(repo.clone());
        let tokenizer_path = api.get("tokenizer.json")?;

        tokenizers::Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
    }

    fn model(&self) -> anyhow::Result<std::path::PathBuf>{
        let api = hf_hub::api::sync::Api::new()?;

        let repo = String::from("Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF");
        let filename = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";


        let model_path = api.repo(Repo::with_revision(
            repo, hf_hub::RepoType::Model, "main".to_string()
        )).get(filename)?;

        
        Ok(model_path)
    }
    //Moved Function To Impl Block From The Original Example
    fn logits_processor(&self) -> LogitsProcessor{
            let temperature = self.temperature;
            //Sampling Pattern
            let sampling = if temperature <= 0. {
                Sampling::ArgMax
            } else {
                match (self.top_k, self.top_p) {
                    (None, None) => Sampling::All { temperature },
                    (Some(k), None) => Sampling::TopK { k, temperature },
                    (None, Some(p)) => Sampling::TopP { p, temperature },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
                }
            };
            LogitsProcessor::from_sampling(self.seed, sampling)
    }
}

fn main() -> anyhow::Result<()>{
    let args = Args::parse();

    let model_path = args.model()?;
    let mut file = File::open(&model_path)?;
    let device = Device::Cpu;
    let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path.clone()))?;

    let mut model = candle_transformers::models::quantized_qwen2::ModelWeights::from_gguf(
        content, 
        &mut file,
        &device
    )?;
    
    let tokenizer = args.tokenizer()?;

    let mut tos = TokenOutputStream::new(tokenizer);
    let prompt_str = args.prompt.clone().expect("Please enter a prompt");
    let prompt_str = format!("<|im_start|>user\n{prompt_str}<|im_end|>\n<|im_start|>assistant\n");

    let tokens= tos
    .tokenizer()
    .encode(prompt_str, true)
    .map_err(anyhow::Error::msg)?;
    let tokens = tokens.get_ids();
    let to_sample = args.sample_len.saturating_sub(1);
    let mut all_tokens: Vec<u32> = vec![];


    let mut next_token = if !args.split_prompt {
        let input = Tensor::new(tokens, &device)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
        let logits = model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        args.logits_processor().sample(&logits)?
    } else {
        let mut next_token = 0;
        for (pos, token) in tokens.iter().enumerate() {
            let input = Tensor::new(&[*token], &device)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
            let logits = model.forward(&input, pos)?;
            let logits = logits.squeeze(0).map_err(anyhow::Error::msg)?;
            next_token = args.logits_processor().sample(&logits)?
        }
        next_token
    };
    all_tokens.push(next_token);
    if let Some(t) = tos.next_token(next_token)? {
        print!("{t}");
        stdout().flush()?;
    }

    let eos_token = "<|im_end|>";
    let eos_token = *tos.tokenizer().get_vocab(true).get(eos_token).unwrap();
    
    let start_post_prompt = Instant::now();
    let mut sampled = 0;
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0).map_err(anyhow::Error::msg)?;
        let logits = model.forward(&input, tokens.len() + index)?;
        let logits = logits.squeeze(0).map_err(anyhow::Error::msg)?;
        let logits = {
            let start_at = all_tokens.len().saturating_sub(args.repeat_last_n);
            apply_repeat_penalty(
                &logits, 
                args.repeat_penalty, 
                &all_tokens[start_at..],
            )?
        };
        next_token = args.logits_processor().sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            stdout().flush()?;
        }
        sampled += 1;
        if next_token == eos_token {
            break;
        };
    }

    if let Some(rest) = tos.decode_rest().map_err(candle_core::Error::msg)? {
        print!("{rest}");
        stdout().flush()?;
    }

    let dt = start_post_prompt.elapsed();
    println!(
        "\n{sampled:4} tokens generated: {:.2} token/s",
        sampled as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
