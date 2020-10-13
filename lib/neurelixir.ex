defmodule Neurelixir do
  def start(x, y, z) do
    #mandar entrada para camadas
    layers_result = send_to_hidden_layers(x, y, z)

    # Chama duas camadas de saída
    [
    layers_result |> output_layer |> Float.round() ,
    layers_result |> output_layer |> Float.round()
    ]
  end

  def send_to_hidden_layers(x, y, z) do
    hidden_layer(x, y, z)
  end

  def hidden_layer(x, y, z) do
    %{
      1 => (x * 0.1) + (y * 0.2) + (z * 0.3) |> Float.round(2),
      2 => (x * 0.4) + (y * 0.5) + (z * 0.6) |> Float.round(2),
      3 => (x * 1.1) + (y * 0.3) + (z * 0.4) |> Float.round(2),
      4 => (x * 0.7) + (y * 0.6) + (z * 0.2) |> Float.round(2)
    }
  end

  def output_layer(list) do
    # Pega cada resultado de hidden layer e adiciona um peso aleatório, depois soma os resultados para definir camada de saida
    Enum.map(
      list,
      fn x ->
        rand = Enum.random(0..100) /100
        elem(x, 1) * rand
      end
    ) |> Enum.sum()
  end
end

#Enviar entradas
Neurelixir.start(1,2,3) |> IO.inspect()
