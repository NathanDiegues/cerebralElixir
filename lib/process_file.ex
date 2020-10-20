defmodule ProcessFile do
  def start do
    read("/home/local/IDEALINVEST/ronney.chaves/Workspaces/elixir/neurelixir/files/dataset.csv")
    # read("/home/local/IDEALINVEST/ronney.chaves/Workspaces/elixir/neurelixir/files/mnist_train.csv")

  end

  def read(filename) do

    File.stream!(filename)
    |> Stream.map(&String.trim/1)
    |> Stream.with_index
    |> Stream.map(fn ({line, _}) -> process_line(line) end)
    |> Stream.run
  end

  def process_line(line) do
    String.split(line,",")
    |> Enum.map(&String.to_integer/1)
    |> IO.inspect()
    # IO.puts "#{line}"
  end

end
