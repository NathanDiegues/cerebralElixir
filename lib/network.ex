defmodule Network do
  @moduledoc """
  Documentation for `Network`.
  """
  defstruct [
    :inputs,
    :hiddens,
    :outputs,
    :hidden_weights,
    :output_weights,
    :learning_rate
  ]

  @doc """
  Hello world.

  ## Examples

      iex> CerebralElixir.hello()
      :world

  """
  def create(inputs, hiddens, outputs, learning_rate) do
    %Network{
      inputs: inputs,
      hiddens: hiddens,
      outputs: outputs,
      learning_rate: learning_rate,
      hidden_weights: Matrex.random(hiddens,inputs),
      output_weights: Matrex.random(outputs,hiddens)
    }
  end

  @spec predict(Network, any) :: any
  def predict(network, input_data) do

    ## forward propagation
    hidden_inputs = Matrex.dot(network.hidden_weights, Matrex.new(input_data)) |> Matrex.apply(&sigmoid/1)
    # apply sigmoid
    Matrex.dot(network.output_weights, hidden_inputs)
    # apply sigmoid
  end

  def sigmoid(value) do
    1.0 / (1 + :math.exp(-1*value))
  end

end
