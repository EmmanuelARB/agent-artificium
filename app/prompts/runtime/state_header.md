[SYSTEM STATE]
Timestamp: {{timestamp}}
Wake reason: {{wake_reason}}
Engine: {{engine_name}}
Working-memory target: {{working_memory_tokens}} tokens; model serving context: {{context_window_tokens}} tokens
Working context estimate: approximately {{context_tokens}} tokens ({{context_percent}}% of working-memory target)
This header is a local estimate. The harness uses a provider preflight count when available to enforce offloading and generation budgets for the complete request.
Tokens added since last context notice: {{tokens_since_last_notice}}
Context status: {{context_status}}
Pending interaction focus: {{current_interaction_or_none}}
Pending event count: {{pending_event_count}}
Active Infinite Attention stream: {{active_stream_or_none}}
Pre-sleep reflection issued this wake: {{pre_sleep_issued}}
Last working-memory-offload checkpoint: {{last_checkpoint_or_none}}
Meta-memory: approximately {{meta_memory_tokens}} tokens / {{meta_memory_words}} words
Meta-memory Guidance Notification threshold: {{meta_memory_guidance_tokens}} tokens
Configured native image mode: {{vision_mode}}
Native multimodal transport: images only
Visual-input guidance: {{vision_guidance}}
Active visual context: {{active_image_count}} image(s)
Loaded images: {{active_images_or_none}}

This is the current state of your life-loop. One-shot images disappear after a
successful inference. Persistent images are transmitted again on every
inference until you call `release_images` or complete working-memory offloading;
release them when repeated visual access no longer justifies the cost.
