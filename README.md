# GenerationPipeline

Repo to contain code for the generation pipeline for synthetic data generation for multimodal LLMs.

## Overview

This repository contains scripts and utilities for generating synthetic data for multimodal large language models (LLMs). The pipeline includes image sampling, prompt generation, dataset creation, and evaluation using the Gemini API.

## Directory Structure
GenerationPipeline/
├── EvaluationPipeline/
│ ├── evaluation.py
│ └── main.py
├── GenerationPipeline/
│ ├── create_dataset.py
│ ├── image_sampling.py
│ ├── llm_prompt.py
│ └── main.py
├── Utils/
├── json_utils.py
├── merge_json_datasets.py
└── README.md
└── requirements.txt


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/GenerationPipeline.git
    cd GenerationPipeline
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    - Create a [.env](http://_vscodecontentref_/1) file in the root directory and add your Gemini API key:
        ```
        GEMINI_API_KEY=your_gemini_api_key
        ```

## Usage

### Image Sampling and Prompt Generation

The main script for image sampling and prompt generation is `main.py` in the `GenerationPipeline` directory. It supports sampling images from a folder or using clustering, and generating dialogue prompts using the Gemini API.

#### Command-line Arguments

- `--num_datapoints`: Number of datapoints to generate (default: 200).
- `--images_file_path`: Path to the images folder or dataset.
- `--save_to`: Name of the output JSON file (default: `output.json`).
- `--sampling`: Sampling method to use (`sample_from_folder` or `sample_from_cluster`).
- `--num_clusters`: Number of clusters (required if sampling from cluster).

#### Example

```sh
python GenerationPipeline/main.py --num_datapoints 100 --images_file_path ./images --save_to output.json --sampling sample_from_folder
```

### Merging JSON Datasets
The script merge_json_datasets.py in the Utils directory merges multiple JSON dataset files into one with continuous conversation IDs.

#### Command-line Arguments
- `--input_files`: List of JSON files to merge.
- `--output_file`: Output JSON file path.

#### Example
```sh
python Utils/merge_json_datasets.py --input_files file1.json file2.json --output_file merged_output.json
```
### Evaluation
The evaluation script evaluation.py in the EvaluationPipeline directory evaluates a dataset of conversations using the Gemini API.

#### Command-line Arguments
- `--dataset_path`: Path to the JSON dataset file.
- `--output_path`: Path to the output file where evaluated scores will be saved (default: evaluation_results.json).

#### Example
```sh
python EvaluationPipeline/main.py --dataset_path dataset.json --output_path evaluation_results.json
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License.