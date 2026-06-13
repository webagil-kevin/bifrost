package anthropic

import (
	"testing"
	"time"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
)

// rawAdvisorResponse mirrors the user-reported Anthropic response: an assistant
// turn containing server_tool_use(advisor) + advisor_tool_result + text.
const rawAdvisorResponse = `{
  "model": "claude-sonnet-4-6",
  "id": "msg_01V5B47VyNHS4XfYEvUEgbBs",
  "type": "message",
  "role": "assistant",
  "content": [
    { "type": "server_tool_use", "id": "srvtoolu_01WJ", "name": "advisor", "input": {} },
    {
      "type": "advisor_tool_result",
      "tool_use_id": "srvtoolu_01WJ",
      "content": { "type": "advisor_result", "text": "Use a channel-based coordination pattern." }
    },
    { "type": "text", "text": "Here is the implementation." }
  ],
  "stop_reason": "end_turn",
  "usage": { "input_tokens": 2733, "output_tokens": 3909 }
}`

// TestAdvisorResponse_RoundTripPreservesBlocks is the regression test for the
// user-reported bug: Anthropic -> Bifrost -> Anthropic dropped the advisor
// server_tool_use and advisor_tool_result blocks, leaving only the text block.
func TestAdvisorResponse_RoundTripPreservesBlocks(t *testing.T) {
	var resp AnthropicMessageResponse
	if err := sonic.Unmarshal([]byte(rawAdvisorResponse), &resp); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}

	ctx := schemas.NewBifrostContext(nil, time.Time{})

	bifrostResp := resp.ToBifrostResponsesResponse(ctx)
	if bifrostResp == nil {
		t.Fatal("ToBifrostResponsesResponse returned nil")
	}

	// Neutral layer must carry an advisor_call with the advice text.
	var foundAdvisor bool
	for _, out := range bifrostResp.Output {
		if out.Type != nil && *out.Type == schemas.ResponsesMessageTypeAdvisorCall {
			foundAdvisor = true
			if out.ResponsesToolMessage == nil || out.ResponsesToolMessage.ResponsesAdvisorCall == nil {
				t.Fatal("advisor_call present but ResponsesAdvisorCall payload missing")
			}
			adv := out.ResponsesToolMessage.ResponsesAdvisorCall
			if adv.ResultType != "advisor_result" {
				t.Errorf("advisor result_type = %q, want advisor_result", adv.ResultType)
			}
			if adv.Text == nil || *adv.Text != "Use a channel-based coordination pattern." {
				t.Errorf("advisor text not carried: %+v", adv.Text)
			}
		}
	}
	if !foundAdvisor {
		t.Fatal("neutral output has no advisor_call message — advisor blocks dropped on parse")
	}

	// Convert back to Anthropic and confirm all three blocks survive in order.
	back := ToAnthropicResponsesResponse(ctx, bifrostResp)
	if back == nil {
		t.Fatal("ToAnthropicResponsesResponse returned nil")
	}
	out, err := sonic.Marshal(back)
	if err != nil {
		t.Fatalf("marshal back: %v", err)
	}

	blocks := gjson.GetBytes(out, "content")
	var types []string
	var advisorText, serverToolName, advisorToolUseID string
	blocks.ForEach(func(_, b gjson.Result) bool {
		bt := b.Get("type").String()
		types = append(types, bt)
		switch bt {
		case "server_tool_use":
			serverToolName = b.Get("name").String()
		case "advisor_tool_result":
			advisorToolUseID = b.Get("tool_use_id").String()
			// inner content may be object or single-element array
			advisorText = b.Get("content.text").String()
			if advisorText == "" {
				advisorText = b.Get("content.0.text").String()
			}
		}
		return true
	})

	if len(types) != 3 || types[0] != "server_tool_use" || types[1] != "advisor_tool_result" || types[2] != "text" {
		t.Fatalf("converted content blocks = %v, want [server_tool_use advisor_tool_result text]\n%s", types, out)
	}
	if serverToolName != "advisor" {
		t.Errorf("server_tool_use name = %q, want advisor", serverToolName)
	}
	if advisorToolUseID != "srvtoolu_01WJ" {
		t.Errorf("advisor_tool_result tool_use_id = %q, want srvtoolu_01WJ", advisorToolUseID)
	}
	if advisorText != "Use a channel-based coordination pattern." {
		t.Errorf("advisor text not preserved through round-trip: %q", advisorText)
	}
}
