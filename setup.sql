
CREATE TABLE public.webcam_detections (
    webcam uuid NOT NULL,
    date timestamp without time zone NOT NULL,
    detections integer
);

CREATE TABLE public.webcam_urls (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    url character varying,
    name character varying,
    debug boolean DEFAULT false NOT NULL,
);

ALTER TABLE ONLY public.webcam_detections
    ADD CONSTRAINT webcam_detections_un UNIQUE (webcam, date);

