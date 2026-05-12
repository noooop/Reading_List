# Observability

随着系统变得越来越复杂，以及越来越多的利用多线程、多进程、分布式，知道系统是否已预期的方式执行变的越来越困难。

## Overview

Observability 一般包括 链路traces、 指标metrics 和 日志logs。

你需要维护的东西大致可以分成系统和软件：
- 系统是指更新频率低，比较稳定的东西，你相信它一直是运行正常的，一般有指标metrics 和 日志 logs就可以确认它是否工作正常。
- 软件是指更新频率高，经常变化的东西，你需要通过链路traces，确认它的内部状态是否正常。

我个人把链路traces 当做轻量级自定义的profiler （比如 pytorch profiler）大致定位异常，
比如io操作（下载图片）是否超时，可以进一步压缩的overhead，可以避免的同步操作，以及调度和执行是否流畅。

## OpenTelemetry, also known as OTel

OpenTelemetry 可以一次instrumentation，随意切换后端。已经成为 Observability 事实上的标准。

## Instrumentation (打点, 插桩)

Instrumentation is the act of adding observability code to an app yourself.

### Zero-code Instrumentation

这种方式只简单的在分布式组建边缘 （比如rpc或者http调用）打点，功能比较局限。

## Manual Instrument

手动打点。

# OpenTelemetry docs

https://opentelemetry.io/docs/


## Traces

The path of a request through your application.

### Tracer Provider

A Tracer Provider (sometimes called TracerProvider) is a factory for Tracers. In most applications, a Tracer Provider is initialized once and its lifecycle matches the application’s lifecycle.

- resource: Optional[Resource] = None
- active_span_processor: Union[SynchronousMultiSpanProcessor, ConcurrentMultiSpanProcessor, None] = None

#### Resource 

A Resource is an immutable representation of the entity producing telemetry as Attributes.

- attributes: Optional zero or more key-value pairs.
- schema_url: Optional URL pointing to the schema.
- resource_detectors

#### Exporter 

(opentelemetry-exporter-otlp)

- OTLPGrpcExporter
- OTLPHttpExporter

trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))

### Tracer

A Tracer creates spans containing more information about what is happening for a given operation, such as a request in a service. Tracers are created from Tracer Providers.

### Context Propagation

#### Span Context

The state of a Span to propagate between processes.

- trace_id: The ID of the trace that this span belongs to.
- span_id: This span's ID.
- is_remote: True if propagated from a remote parent.
- trace_flags: Trace options to propagate.
- trace_state: Tracing-system-specific info to propagate.

#### Propagation

(opentelemetry-api)

TraceContextTextMapPropagator (Extracts and injects using w3c TraceContext's headers.)

- inject
- extract
- fields

#### Baggage

Baggage allows you to propagate arbitrary key-value pairs.

### Spans

A span represents a unit of work or operation. Spans are the building blocks of Traces.

One way to think of Traces is that they’re a collection of structured logs with context, correlation, hierarchy, and more baked in.

- trace_id
- span_id
- parent_id

#### ReadableSpan

Provides read-only access to span attributes.

name: The name of the operation this span represents
context: The immutable span context
parent: This span's parent's `opentelemetry.trace.SpanContext`, or
    None if this is a root span
resource: Entity producing telemetry
attributes: The span's attributes to be exported
events: Timestamped events to be exported
links: Links to other spans to be exported

kind: trace_api.SpanKind = trace_api.SpanKind.INTERNAL
instrumentation_info: Optional[InstrumentationInfo] = None
status: Status = Status(StatusCode.UNSET)
start_time: Optional[int] = None
end_time: Optional[int] = None
instrumentation_scope: Optional[InstrumentationScope] = None