# Surrogate Modeling Framework

This repository serves as a base for developing and experimenting across the surrogate model ecosystem, particularly neural network-based surrogate models for partial differential equations (PDEs). The framework includes numerical solvers, neural network architectures, training utilities, and uncertainty quantification (UQ) tools to facilitate the development and analysis of Neural-PDE surrogate models.

## Features

- **Numerical Solvers**: Implements various numerical solvers for PDEs, providing a foundation for generating training data and validating surrogate models.
- **Neural Network Surrogate Models**: Provides a collection of neural network architectures specifically designed for surrogate modeling of PDEs.
- **Training Utilities**: Offers a set of utilities to streamline the training process, including data preprocessing, model training, and evaluation.
- **Uncertainty Quantification**: Includes tools for quantifying and analyzing the uncertainty associated with surrogate models, enabling robust decision-making.
- **Active Learning**: (TBD) Implements active learning strategies to efficiently acquire informative training data, reducing the computational cost and improving model accuracy.
- **Experimental Framework**: Serves as a base repository for conducting experiments and developing new methods in the surrogate modeling domain.

## Getting Started

To get started with the Surrogate Modeling Framework, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/Neural-PDEs.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Explore the available numerical solvers, neural network architectures, and training utilities in the respective directories.
4. Refer to the documentation and examples for guidance on using the framework components.

## Experimental Workflow

The Surrogate Modeling Framework enables researchers and practitioners to conduct experiments and develop new methods for surrogate modeling. The typical workflow includes:

1. **Data Generation**: Use the numerical solvers to generate training and validation data for the surrogate models.
2. **Model Development**: Develop and implement new neural network architectures or adapt existing ones for surrogate modeling of PDEs.
3. **Training and Evaluation**: Utilize the training utilities to train the surrogate models and evaluate their performance using appropriate metrics.
4. **Uncertainty Quantification**: Apply the UQ tools to assess the uncertainty associated with the surrogate models and make informed decisions.
5. **Active Learning**: Employ active learning strategies to efficiently acquire informative training data and improve model accuracy.
6. **Experimentation**: Conduct experiments to compare different surrogate modeling approaches, investigate the impact of various factors, and develop new methods.

## Contributing

We welcome contributions to the Neural-PDE Surrogate Modeling Framework! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or inquiries, please contact the project maintainer at vignesh7g@gmail.com

Happy surrogate modeling!
