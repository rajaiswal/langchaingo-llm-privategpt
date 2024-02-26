# langchaingo-llm-privategpt
langchaingo extension to use privateGPT as an LLM

Installation (Optional step for generation)
```shell
    go get github.com/ogen-go/ogen/gen@v0.82.0
```

```shell
    go generate ./...
```

Usage

```go
    // Initialize LLM
    llm, err := privategpt.New(
        privategpt.WithServerURL("http://localhost:8001"),
        privategpt.WithSystemPrompt("This is system prompt"),
    )
    if err != nil {
        panic(err)
    }

    // Generate content
    res, err := llm.GenerateContent(
        context.Background(),
        []llms.MessageContent{
            llms.TextParts(schema.ChatMessageTypeGeneric, "This is a prompt"),
        },
    )
    if err != nil {
        panic(err)
    }
```

