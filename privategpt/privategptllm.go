package privategpt

import (
	"context"
	"errors"
	"privateGPTGo/llms/privategpt/internal/privategptclient"

	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

var (
	ErrEmptyResponse       = errors.New("no response")
	ErrIncompleteEmbedding = errors.New("no all input got emmbedded")
)

// LLM is a privategpt LLM implementation.
type LLM struct {
	CallbacksHandler callbacks.Handler
	client           *privategptclient.Client
	options          options
}

var _ llms.Model = (*LLM)(nil)

// New creates a new ollama LLM implementation.
func New(opts ...Option) (*LLM, error) {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}

	client, err := privategptclient.NewClient(o.privategptServerURL.String(), o.privategptOptions...)
	if err != nil {
		return nil, err
	}

	return &LLM{client: client, options: o}, nil
}

// Call Implement the call interface for LLM.
func (o *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, o, prompt, options...)
}

// GenerateContent implements the Model interface.
// nolint: goerr113
func (o *LLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) { // nolint: lll, cyclop, funlen
	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMGenerateContentStart(ctx, messages)
	}

	opts := llms.CallOptions{}
	for _, opt := range options {
		opt(&opts)
	}

	// Our input is a sequence of MessageContent, each of which potentially has
	// a sequence of Part that is text.
	// We have to convert it to a format PrivateGPT understands: ChatBody, which
	// has a sequence of Message, each of which has a role and content - single
	// text + potential images.
	chatMsgs := make([]privategptclient.OpenAIMessage, 0, len(messages)+1)
	chatMsgs = append(chatMsgs, privategptclient.OpenAIMessage{
		Role:    privategptclient.NewOptOpenAIMessageRole(privategptclient.OpenAIMessageRoleSystem),
		Content: privategptclient.NewStringOpenAIMessageContent(o.options.system),
	})

	for _, mc := range messages {
		msg := privategptclient.OpenAIMessage{
			Role: privategptclient.NewOptOpenAIMessageRole(typeToRole(mc.Role)),
		}

		// Look at all the parts in mc; expect to find a single Text part and
		// any number of binary parts.
		var text string
		foundText := false

		for _, p := range mc.Parts {
			switch pt := p.(type) {
			case llms.TextContent:
				if foundText {
					return nil, errors.New("expecting a single Text content")
				}
				foundText = true
				text = pt.Text
			case llms.BinaryContent:
				// images = append(images, privategptclient.ImageData(pt.Data))
			default:
				return nil, errors.New("only support Text and BinaryContent parts right now")
			}
		}

		msg.Content = privategptclient.NewStringOpenAIMessageContent(text)
		chatMsgs = append(chatMsgs, msg)
	}

	stream := func(b bool) *bool { return &b }(opts.StreamingFunc != nil)

	req, err := o.client.ChatCompletionV1ChatCompletionsPost(ctx, &privategptclient.ChatBody{
		Messages:       chatMsgs,
		UseContext:     privategptclient.NewOptBool(false),
		IncludeSources: privategptclient.NewOptBool(false),
		Stream:         privategptclient.NewOptBool(*stream),
	})
	if err != nil {
		return nil, err
	}

	res, ok := req.(*privategptclient.OpenAICompletion)
	if !ok {
		return nil, errors.New("unexpected response type")
	}

	choices := []*llms.ContentChoice{}

	for i := range res.Choices {
		choice := res.Choices[i]

		choices = append(choices, &llms.ContentChoice{
			Content: choice.GetMessage().Value.OpenAIMessage.Content.String,
			// GenerationInfo: map[string]any{
			// 	"CompletionTokens": resp.EvalCount,
			// 	"PromptTokens":     resp.PromptEvalCount,
			// 	"TotalTokens":      resp.EvalCount + resp.PromptEvalCount,
			// },
			StopReason: choice.GetFinishReason().String,
		})
	}

	response := &llms.ContentResponse{Choices: choices}

	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMGenerateContentEnd(ctx, response)
	}

	return response, nil
}

func (o *LLM) CreateEmbedding(ctx context.Context, inputTexts []string) ([][]float32, error) {
	embeddings := [][]float32{}

	res, err := o.client.EmbeddingsGenerationV1EmbeddingsPost(ctx, &privategptclient.EmbeddingsBody{
		Input: privategptclient.EmbeddingsBodyInput{
			Type:        "string",
			StringArray: inputTexts,
		},
	})
	if err != nil {
		return nil, err
	}

	er, ok := res.(*privategptclient.EmbeddingsResponse)
	if !ok {
		return nil, errors.New("unexpected response type")
	}

	if len(er.Data[0].Embedding) == 0 {
		return nil, ErrEmptyResponse
	}

	for _, embedding := range er.Data {
		// Convert []float64 to []float32
		sliceFloat32 := make([]float32, len(embedding.Embedding))
		for i, v := range embedding.Embedding {
			sliceFloat32[i] = float32(v)
		}

		embeddings = append(embeddings, sliceFloat32)
	}

	if len(inputTexts) != len(embeddings) {
		return embeddings, ErrIncompleteEmbedding
	}

	return embeddings, nil
}

func typeToRole(typ schema.ChatMessageType) privategptclient.OpenAIMessageRole {
	switch typ {
	case schema.ChatMessageTypeSystem:
		return privategptclient.OpenAIMessageRoleSystem
	case schema.ChatMessageTypeAI:
		return privategptclient.OpenAIMessageRoleAssistant
	case schema.ChatMessageTypeHuman:
		fallthrough
	case schema.ChatMessageTypeGeneric:
		return privategptclient.OpenAIMessageRoleUser
	case schema.ChatMessageTypeFunction:
		return "function"
	}
	return ""
}
