defmodule NeurelixirTest do
  use ExUnit.Case
  doctest Neurelixir

  test "greets the world" do
    assert Neurelixir.hello() == :world
  end
end
