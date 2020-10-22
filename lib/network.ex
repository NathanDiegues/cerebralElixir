defmodule Network do
  @moduledoc """
  Documentation for `Network`.
  """
  @enforce_keys [:inputs, :hiddens, :outputs]
  defstruct [
    :inputs,
    :hiddens,
    :outputs,
    :hidden_weights,
    :output_weights,
    :learning_rate
  ]

  @type t() :: %__MODULE__{
          inputs: integer(),
          hiddens: integer(),
          outputs: integer(),
          hidden_weights: Matrex.t(),
          output_weights: Matrex.t(),
          learning_rate: integer()
        }

  @spec create(pos_integer, pos_integer, pos_integer, any) :: Network.t()
  def create(inputs, hiddens, outputs, learning_rate) do
    %Network{
      inputs: inputs,
      hiddens: hiddens,
      outputs: outputs,
      learning_rate: learning_rate,
      hidden_weights: Matrex.random(hiddens, inputs),
      output_weights: Matrex.random(outputs, hiddens)
    }
  end

  @spec predict(Network, list()) :: any
  def predict(network, input_data) do
    ## forward propagation
    hidden_outputs =
      Matrex.dot(network.hidden_weights, Matrex.new(to_lol(input_data)))
      |> Matrex.apply(&sigmoid/1)

    Matrex.dot(network.output_weights, hidden_outputs) |> Matrex.apply(&sigmoid/1)
  end

  @spec train(Network, list(), list(), integer()) :: any
  def train(network,_,_,0) do
    network
  end

  def train(network, input_data, target_data, epochs) do

    ## forward propagation
    inputs = Matrex.new(to_lol(input_data))
    hidden_outputs = Matrex.dot(network.hidden_weights, inputs) |> Matrex.apply(&sigmoid/1)
    final_outputs = Matrex.dot(network.output_weights, hidden_outputs) |> Matrex.apply(&sigmoid/1)

    ## find errors
    targets = Matrex.new(to_lol(target_data))
    output_errors = Matrex.subtract(targets, final_outputs)
    hidden_errors = Matrex.dot_tn(network.output_weights, output_errors)

    ## backpropagate
    network = %{
      network
      | output_weights:
          Matrex.multiply(output_errors, sigmoidPrime(final_outputs))
          |> Matrex.dot_nt(hidden_outputs)
          |> Matrex.multiply(network.learning_rate)
          |> Matrex.add(network.output_weights)
    }

    %{
      network
      | hidden_weights:
          Matrex.multiply(hidden_errors, sigmoidPrime(hidden_outputs))
          |> Matrex.dot_nt(inputs)
          |> Matrex.multiply(network.learning_rate)
          |> Matrex.add(network.hidden_weights)
    }

    train(network, input_data, target_data, epochs - 1)
  end

  @spec to_lol(list()) :: list(list())
  defp to_lol(list) do
    Enum.map(list, fn x -> [x] end)
  end

  @spec sigmoid(number) :: float
  defp sigmoid(value) do
    1.0 / (1 + :math.exp(-1 * value))
  end

  @spec sigmoidPrime(Matrex.t()) :: Matrex.t()
  defp sigmoidPrime(m) do
    rows = m[:rows]

    ones = Matrex.ones(rows, 1)

    Matrex.multiply(m, Matrex.subtract(ones, m))
  end
end
